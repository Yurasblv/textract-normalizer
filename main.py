import json
import logging
import os
import re
import time
from dataclasses import asdict, field , dataclass
from datetime import date, datetime
from pathlib import Path
from statistics import mean
from typing import Optional

import boto3
from dotenv import load_dotenv

load_dotenv(".env")

out_dir = Path(__file__).parent / "out"
out_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler("out/run.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

textract = boto3.client(
    "textract",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

@dataclass
class PrescriptionMedication:
    drug_name: Optional[str] = None
    dosage_text: Optional[str] = None
    frequency_text: Optional[str] = None
    duration_days: Optional[int] = None
    quantity: Optional[int] = None

@dataclass
class Prescription:
    prescription_date: Optional[str] = None
    prescriber_name: Optional[str] = None
    prescriber_id: Optional[str] = None
    language: str = "it"
    medications: list[PrescriptionMedication] = field(default_factory=list)
    notes: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0

@dataclass
class InvoiceLineItem:
    description: Optional[str] = None
    qty: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None

@dataclass
class Invoice:
    invoice_number: Optional[str] = None
    issue_date: Optional[str] = None
    due_date: Optional[str] = None
    supplier_name: Optional[str] = None
    currency: Optional[str] = None
    invoice_total: Optional[float] = None
    line_items: list[InvoiceLineItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    quality_score: float = 0.0


class TextractInvoiceParser:
    REQUIRED_FIELDS = ["invoice_number", "issue_date", "supplier_name", "invoice_total"]

    def __init__(self, textract_json: dict):
        self.data = textract_json
        self.lines = self._extract_lines()
        self.invoice = Invoice()
        self._parse_data()

    def _extract_lines(self):
        """Flatten all LINE blocks with text and confidence."""
        lines = []
        for block in self.data.get("Blocks", []):
            if block.get("BlockType") == "LINE":
                lines.append(
                    {
                        "text": block.get("Text", "").strip(),
                        "confidence": block.get("Confidence", 0.0),
                    }
                )
        return lines

    def _parse_data(self):
        confidences = []
        found_fields = 0

        supplier = self._find_supplier_name()
        if supplier:
            self.invoice.supplier_name = supplier["value"]
            confidences.append(supplier["confidence"])
            found_fields += 1

        inv_num = self._find_invoice_number()
        if inv_num:
            self.invoice.invoice_number = inv_num["value"]
            confidences.append(inv_num["confidence"])
            found_fields += 1

        issue_date = self._find_date(r"(data documento|issue date)", fallback=True)
        if issue_date:
            self.invoice.issue_date = issue_date["value"]
            confidences.append(issue_date["confidence"])
            found_fields += 1

        due_date = self._find_date(r"(scadenza|due date)")

        if due_date:
            self.invoice.due_date = due_date["value"]
            confidences.append(due_date["confidence"])

        total = self._find_total()

        if total:
            self.invoice.invoice_total = total["value"]
            self.invoice.currency = total["currency"]
            confidences.append(total["confidence"])
            found_fields += 1

        self.invoice.line_items = self._extract_line_items()
        self.invoice.quality_score = TextractNormalizer.calc_quality_score(
            confidences, found_fields, len(self.REQUIRED_FIELDS)
        )
        self.invoice.warnings = self._collect_warnings(found_fields)

    def _find_supplier_name(self):
        for line in self.lines[:5]:  # usually top of doc
            if re.search(r"(s\.r\.l\.|spa|ltd|inc|marilab|company|sede)", line["text"], re.I):
                return {"value": line["text"], "confidence": line["confidence"]}
        return None

    def _find_invoice_number(self):
        for line in self.lines:
            if re.search(r"(numero documento|fattura n|invoice number|n\.)", line["text"], re.I):
                match = re.search(r"\d{3,}", line["text"])
                if match:
                    return {"value": match.group(0), "confidence": line["confidence"]}
        return None

    def _find_date(self, keyword_pattern, fallback=False):
        for line in self.lines:
            if re.search(keyword_pattern, line["text"], re.I):
                match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", line["text"])
                if match:
                    try:
                        return {
                            "value": datetime.strptime(match.group(1), "%d/%m/%Y").date().strftime(),
                            "confidence": line["confidence"],
                        }
                    except ValueError:
                        pass
        if fallback:
            for line in self.lines:
                match = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", line["text"])
                if match:
                    try:
                        return {
                            "value": datetime.strptime(match.group(1), "%d/%m/%Y").date(),
                            "confidence": line["confidence"],
                        }
                    except ValueError:
                        continue
        return None

    def _find_total(self):
        for line in reversed(self.lines):  # totals usually at bottom
            if re.search(r"(totale|invoice total|importo)", line["text"], re.I):
                match = re.search(r"([€$])?\s?([\d\.,]+)", line["text"])
                if match:
                    currency = match.group(1) if match.group(1) else "EUR"
                    amount = self._safe_float(match.group(2))
                    return {
                        "value": amount,
                        "currency": currency,
                        "confidence": line["confidence"],
                    }
        return None

    def _extract_line_items(self):
        items = []
        block_map = {b["Id"]: b for b in self.data.get("Blocks" , [])}
        for block in self.data.get("Blocks" , []):
            if block.get("BlockType") == "TABLE":
                rows = {}
                for rel in block.get("Relationships" , []):
                    if rel["Type"] == "CHILD":
                        for cid in rel["Ids"]:
                            cell = block_map[cid]
                            if cell["BlockType"] == "CELL":
                                row = cell["RowIndex"]
                                col = cell["ColumnIndex"]
                                text = self._get_text(cell , block_map)
                                rows.setdefault(row , {})[col] = text

                for r in sorted(rows.keys()):
                    row = rows[r]
                    if len(row) >= 4:
                        desc = row.get(1 , "")
                        qty = self._safe_float(row.get(2 , ""))
                        unit_price = self._safe_float(row.get(3 , ""))
                        total = self._safe_float(row.get(4 , ""))
                        items.append(
                            InvoiceLineItem(
                                description=desc , qty=qty ,
                                unit_price=unit_price , total=total
                            )
                        )
        return items

    def _get_text(self , cell , block_map):
        text = []
        for rel in cell.get("Relationships" , []):
            if rel["Type"] == "CHILD":
                for cid in rel["Ids"]:
                    word = block_map[cid]
                    if word.get("BlockType") in ("WORD" , "SELECTION_ELEMENT"):
                        text.append(word.get("Text" , ""))
        return " ".join(text).strip()

    def _collect_warnings(self, found_fields):
        warnings = []
        for f in self.REQUIRED_FIELDS:
            if getattr(self.invoice, f) is None:
                warnings.append(f"Missing required field: {f}")
        if found_fields < len(self.REQUIRED_FIELDS):
            warnings.append("Some key fields not extracted with high confidence")
        return warnings

    def _safe_float(self, val: str) -> Optional[float]:
        try:
            return float(val.replace(",", "."))
        except Exception:
            return None


class TextractPrescriptionParser:
    def __init__(self , textract_json) -> None:
        self.textract_json = textract_json
        self.lines = self._extract_lines()
        self.prescription = Prescription()
        self._parse_data()

    def _extract_lines(self) -> list[str]:
        lines = []
        for block in self.textract_json.get('Blocks' , []):
            if block.get('BlockType') == 'LINE' and 'Text' in block:
                lines.append(block['Text'])
        return lines

    def _parse_date(self , text: str) -> Optional[date]:
        patterns = [
            r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})' ,  # DD/MM/YYYY or DD-MM-YYYY
            r'(\d{1,2})\s+([a-zA-Z]+)\s+(\d{2,4})' ,  # DD Month YYYY (Italian months)
        ]

        for pattern in patterns:
            match = re.search(pattern , text)
            if match:
                try:
                    if '/' in text or '-' in text or '.' in text:
                        day , month , year = match.groups()
                        day , month , year = int(day) , int(month) , int(year)
                        if year < 100:
                            year += 2000 if year < 50 else 1900
                        return date(year , month , day).strftime()
                    else:
                        day , month_str , year = match.groups()
                        day , year = int(day) , int(year)
                        month_map = {
                            'gennaio': 1 , 'febbraio': 2 , 'marzo': 3 , 'aprile': 4 ,
                            'maggio': 5 , 'giugno': 6 , 'luglio': 7 , 'agosto': 8 ,
                            'settembre': 9 , 'ottobre': 10 , 'novembre': 11 , 'dicembre': 12
                        }
                        month = month_map.get(month_str.lower())
                        if month:
                            return date(year , month , day).strftime()
                except (ValueError , TypeError):
                    continue
        return None

    def _parse_quantity(self , text: str) -> Optional[int]:
        patterns = [
            r'qta\s*[:]?\s*(\d+)' ,
            r'n\.\s*(\d+)' ,
            r'quantit[àa]\s*[:]?\s*(\d+)' ,
            r'(\d+)\s*(compresse|cp|fiale|fl|ml|g|mg|µg)' ,
        ]

        for pattern in patterns:
            match = re.search(pattern , text , re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def _parse_duration(self , text: str) -> Optional[int]:
        patterns = [
            r'per\s*(\d+)\s*giorni' ,  # per 7 giorni
            r'(\d+)\s*giorni' ,  # 7 giorni
            r'durata\s*[:]?\s*(\d+)\s*g' ,  # durata: 7g
        ]

        for pattern in patterns:
            match = re.search(pattern , text , re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None

    def _parse_frequency(self , text: str) -> Optional[str]:
        patterns = [
            r'(\d+)\s*volte\s*al\s*giorno' ,  # 2 volte al giorno
            r'(\d+)\s*[xX]\s*die' ,  # 2 x die
            r'ogni\s*(\d+)\s*ore' ,  # ogni 8 ore
            r'(\d+)\s*[cp]\s*al\s*dì' ,  # 2 cp al dì
        ]

        for pattern in patterns:
            match = re.search(pattern , text , re.IGNORECASE)
            if match:
                try:
                    return f"{match.group(1)} volte al giorno"
                except (ValueError , IndexError):
                    continue
        return None

    def _parse_dosage(self , text: str) -> Optional[str]:
        patterns = [
            r'(\d+[\.,]?\d*)\s*(mg|ml|g|µg)' ,  # 5mg, 2.5 ml
            r'dose\s*[:]?\s*(\d+[\.,]?\d*)\s*(mg|ml|g|µg)' ,  # dose: 5mg
        ]

        for pattern in patterns:
            match = re.search(pattern , text , re.IGNORECASE)
            if match:
                try:
                    amount = match.group(1).replace(',' , '.')
                    unit = match.group(2)
                    return f"{amount} {unit}"
                except (ValueError , IndexError):
                    continue
        return None

    def _calculate_quality_score(self) -> float:
        score = 0.0
        total_possible = 5.0

        if self.prescription.prescription_date:
            score += 1.0
        if self.prescription.prescriber_name:
            score += 1.0
        if self.prescription.prescriber_id:
            score += 1.0
        if self.prescription.medications:
            score += 1.0
        if any(m.drug_name for m in self.prescription.medications):
            score += 1.0

        return min(score / total_possible , 1.0)

    def _parse_data(self) -> Prescription:
        for line in self.lines:
            if 'data' in line.lower():
                date_obj = self._parse_date(line)
                if date_obj:
                    self.prescription.prescription_date = date_obj
                    break

        for line in self.lines:
            if any(term in line.lower() for term in ['dott' , 'dr.' , 'medico' , 'farmacista']):
                self.prescription.prescriber_name = line
            if any(term in line.lower() for term in ['codice fiscale' , 'cf:' , 'id']):
                id_match = re.search(r'[A-Z0-9]{11,16}' , line)
                if id_match:
                    self.prescription.prescriber_id = id_match.group(0)

        current_med = None
        for line in self.lines:
            if (
                line.isupper() or
                any(term in line.lower() for term in [
                    'compresse' , 'capsule' , 'fiale' , 'crema' , 'pomata'
                ]
                    )
            ):

                if current_med:
                    self.prescription.medications.append(current_med)

                current_med = PrescriptionMedication(drug_name=line)
            elif current_med:
                if not current_med.dosage_text:
                    current_med.dosage_text = self._parse_dosage(line)
                if not current_med.frequency_text:
                    current_med.frequency_text = self._parse_frequency(line)
                if not current_med.duration_days:
                    current_med.duration_days = self._parse_duration(line)
                if not current_med.quantity:
                    current_med.quantity = self._parse_quantity(line)

        if current_med:
            self.prescription.medications.append(current_med)

        notes = []
        for line in self.lines:
            if (not self.prescription.prescription_date or
                line not in str(self.prescription.prescription_date)) and \
                    (not self.prescription.prescriber_name or
                     line != self.prescription.prescriber_name) and \
                    (not self.prescription.prescriber_id or
                     line != self.prescription.prescriber_id) and \
                    not any(line == m.drug_name for m in self.prescription.medications if m.drug_name):
                notes.append(line)

        if notes:
            self.prescription.notes = " ".join(notes)

        if not self.prescription.prescription_date:
            self.prescription.warnings.append("Data della prescrizione non trovata")
        if not self.prescription.prescriber_name:
            self.prescription.warnings.append("Nome del prescrittore non trovato")
        if not self.prescription.medications:
            self.prescription.warnings.append("Nessun farmaco trovato nella prescrizione")

        self.prescription.quality_score = self._calculate_quality_score()

        return self.prescription


class TextractNormalizer:
    def __init__(self) -> None:
        self.data_dir = Path(__file__).parent / "data"

    @staticmethod
    def calc_quality_score(textract_confidences, required_fields_found, total_required):
        tex_conf = mean(textract_confidences) if textract_confidences else 0
        coverage = required_fields_found / total_required if total_required else 0
        validation = 1 if coverage == 1 else 0.5
        score = 0.4 * tex_conf + 0.4 * coverage + 0.2 * validation
        return round(score, 2)

    @staticmethod
    def normalize_invoice(textract_json: dict) -> Invoice:
        normalizer = TextractInvoiceParser(textract_json)
        return normalizer.invoice

    @staticmethod
    def normalize_prescription(textract_json):
        normalizer = TextractPrescriptionParser(textract_json)
        return normalizer.prescription

    @staticmethod
    def save_json(data, filename):
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            breakpoint()
            f.write(json.dumps(asdict(data), ensure_ascii=False, indent=4))

        logger.info(f"Saved normalized data to {out_path}")

    @staticmethod
    def process_file(
        file_path, file_bytes: bytes, max_retries: int = 3, base_backoff: int = 2
    ) -> None:
        for attempt in range(1, max_retries + 1):
            try:
                filename = file_path.name.lower()

                logger.info(f"Submit attempt {attempt} for {file_path=}")
                response = textract.analyze_document(
                    Document={"Bytes": file_bytes}, FeatureTypes=["TABLES"]
                )

                if "invoice" in filename:
                    data = TextractNormalizer.normalize_invoice(response)
                    TextractNormalizer.save_json(data, "invoice_a.json")

                elif "prescription" in filename:
                    data = TextractNormalizer.normalize_prescription(response)
                    TextractNormalizer.save_json(data, "rx_it.json")

                logger.info(
                    f"Successfully processed {filename=} on {attempt=}"
                )

                return None
            except Exception as e:
                logger.warning(f"Attempt {attempt} failed: {str(e)}, {file_path=}")
                if attempt < max_retries:
                    sleep_time = base_backoff**attempt
                    logger.info(f"Backing off for {sleep_time}s before retry")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for {file_path=}")
                    raise e

    def run(self):
        for file_path in self.data_dir.glob("*.*"):
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                if file_path.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg"]:
                    self.process_file(file_path, file_bytes)
                else:
                    raise Exception(
                        f"Unsupported file type: {file_path.suffix.lower()}"
                    )
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")


if __name__ == "__main__":
    TextractNormalizer().run()
