"""Hand-authored challenge set: fictional documents with manually written labels.

Each case targets one failure mode (negation, uncertainty, old prescriptions,
units the model must not convert, decimal commas, flags between value and
unit, distractor dates, allergies, other people's medicines, empty results).
These are deliberately constructed and are labelled `challenge: true`.
Labels were written and checked by hand against the rules in
shared/extraction_prompt.txt; the script below additionally verifies that
every labelled value occurs in its evidence and the evidence in the text.

  python pipeline/challenge/challenge_set.py   -> data/processed/challenge.jsonl
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from render import Doc, check_gold  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def L(test, value, unit, rr, date, ev):
    return {"test": test, "value": value, "unit": unit, "reference_range": rr, "date": date, "evidence": ev}


def M(name, dose, freq, status, ev):
    return {"name": name, "dose": dose, "frequency": freq, "status": status, "evidence": ev}


def FU(instr, ev):
    return {"instruction": instr, "evidence": ev}


def D(value, ev):
    return {"value": value, "evidence": ev}


CASES = [
    ("negated-medication", """Clinic note - Ashford Lane Surgery
Date: 2024-09-12
Patient: Jonah Pell (fictional)

Current medications: amlodipine 5 mg once daily.
Patient denies taking aspirin.
""", D("2024-09-12", "Date: 2024-09-12"), [], [
        M("amlodipine", "5 mg", "once daily", "current", "Current medications: amlodipine 5 mg once daily."),
        M("aspirin", None, None, "not_current", "Patient denies taking aspirin."),
    ], []),

    ("uncertain-medication", """Telephone note, 3 March 2025
Spoke with Ms. Odile Barrow (fictional). She thinks she may still be using her salbutamol inhaler but was not sure.
She takes sertraline 50 mg every morning.
""", D("3 March 2025", "Telephone note, 3 March 2025"), [], [
        M("salbutamol", None, None, "unclear", "She thinks she may still be using her salbutamol inhaler but was not sure."),
        M("sertraline", "50 mg", "every morning", "current", "She takes sertraline 50 mg every morning."),
    ], []),

    ("old-prescription", """Records summary for Mr. Tomas Rehn (fictional)
Prepared 18/06/2025

Prescription dated 12/03/2019: Amoxicillin 500 mg three times daily for 7 days.
""", D("18/06/2025", "Prepared 18/06/2025"), [], [
        M("Amoxicillin", "500 mg", "three times daily", "unclear", "Prescription dated 12/03/2019: Amoxicillin 500 mg three times daily for 7 days."),
    ], []),

    ("less-than-value-and-pending", """WESTBROOK PATHOLOGY (fictional)
Report date: 7 Jan 2025
C-reactive protein: <0.5 mg/L (ref <5.0)
Ferritin: 41 ug/L (ref 15-200)
Vitamin D: pending
""", D("7 Jan 2025", "Report date: 7 Jan 2025"), [
        L("C-reactive protein", "<0.5", "mg/L", "<5.0", None, "C-reactive protein: <0.5 mg/L (ref <5.0)"),
        L("Ferritin", "41", "ug/L", "15-200", None, "Ferritin: 41 ug/L (ref 15-200)"),
    ], [], []),

    ("vitals-not-labs", """Nurse visit 2025-02-20
Patient: Ida Marsh (fictional)
Blood pressure 142/88 mmHg. Weight 92.4 kg. BMI 31.2 kg/m2. Temperature 36.8 C.
Hemoglobin A1c 7.4 % (lab, collected today).
""", D("2025-02-20", "Nurse visit 2025-02-20"), [
        L("Hemoglobin A1c", "7.4", "%", None, None, "Hemoglobin A1c 7.4 % (lab, collected today)."),
    ], [], []),

    ("distractor-dates", """FERNHILL LABS (fictional)
Patient: Ravi Anand   Date of birth: 04/11/1958
Previous report: 01/08/2023
Collected: 02/10/2024   Report date: 02/12/2024
Sodium     139   mmol/L   135-145
Potassium  4.4   mmol/L   3.5-5.1
""", D("02/12/2024", "Collected: 02/10/2024   Report date: 02/12/2024"), [
        L("Sodium", "139", "mmol/L", "135-145", None, "Sodium     139   mmol/L   135-145"),
        L("Potassium", "4.4", "mmol/L", "3.5-5.1", None, "Potassium  4.4   mmol/L   3.5-5.1"),
    ], [], []),

    ("repeated-measurement-prose", """Diabetes review - 14 November 2023
Her HbA1c was 7.1% in March 2023 and 6.8% on 2 October 2023.
She continues on metformin 1000 mg twice daily.
Repeat HbA1c in 6 months.
""", D("14 November 2023", "Diabetes review - 14 November 2023"), [
        L("HbA1c", "7.1", "%", None, "March 2023", "Her HbA1c was 7.1% in March 2023 and 6.8% on 2 October 2023."),
        L("HbA1c", "6.8", "%", None, "2 October 2023", "Her HbA1c was 7.1% in March 2023 and 6.8% on 2 October 2023."),
    ], [
        M("metformin", "1000 mg", "twice daily", "current", "She continues on metformin 1000 mg twice daily."),
    ], [FU("Repeat HbA1c in 6 months", "Repeat HbA1c in 6 months.")]),

    ("si-units-not-converted", """Kidney function - Harrow Vale Clinic (fictional)
Date: 2025-04-01
Creatinine 88 umol/L (60-110)
eGFR 74 mL/min/1.73m2 (>60)
""", D("2025-04-01", "Date: 2025-04-01"), [
        L("Creatinine", "88", "umol/L", "60-110", None, "Creatinine 88 umol/L (60-110)"),
        L("eGFR", "74", "mL/min/1.73m2", ">60", None, "eGFR 74 mL/min/1.73m2 (>60)"),
    ], [], []),

    ("no-unit-printed", """Thyroid result, 9 Sept 2024
TSH 2.1 (0.4-4.0)
Free T4 14 pmol/L (10-22)
""", D("9 Sept 2024", "Thyroid result, 9 Sept 2024"), [
        L("TSH", "2.1", None, "0.4-4.0", None, "TSH 2.1 (0.4-4.0)"),
        L("Free T4", "14", "pmol/L", "10-22", None, "Free T4 14 pmol/L (10-22)"),
    ], [], []),

    ("frequency-abbreviations", """Medication reconciliation 2025-01-15
Current:
- metformin 500 mg BD
- atorvastatin 20 mg nocte
- paracetamol 1 g PRN
""", D("2025-01-15", "Medication reconciliation 2025-01-15"), [], [
        M("metformin", "500 mg", "BD", "current", "- metformin 500 mg BD"),
        M("atorvastatin", "20 mg", "nocte", "current", "- atorvastatin 20 mg nocte"),
        M("paracetamol", "1 g", "PRN", "current", "- paracetamol 1 g PRN"),
    ], []),

    ("switched-medication", """Letter dated 21 May 2025
Lisinopril was discontinued because of a persistent cough. He was switched to losartan 50 mg daily.
Please check potassium in 2 weeks.
""", D("21 May 2025", "Letter dated 21 May 2025"), [], [
        M("Lisinopril", None, None, "not_current", "Lisinopril was discontinued because of a persistent cough."),
        M("losartan", "50 mg", "daily", "current", "He was switched to losartan 50 mg daily."),
    ], [FU("check potassium in 2 weeks", "Please check potassium in 2 weeks.")]),

    ("negated-follow-up", """Discharge note - 30 Jun 2025
Wound healed well. No further follow-up is needed.
""", D("30 Jun 2025", "Discharge note - 30 Jun 2025"), [], [], []),

    ("two-follow-ups", """Visit summary 2024-11-05
Total cholesterol 6.4 mmol/L.
Repeat lipid panel in 3 months.
Referral to dietitian.
""", D("2024-11-05", "Visit summary 2024-11-05"), [
        L("Total cholesterol", "6.4", "mmol/L", None, None, "Total cholesterol 6.4 mmol/L."),
    ], [], [
        FU("Repeat lipid panel in 3 months", "Repeat lipid panel in 3 months."),
        FU("Referral to dietitian", "Referral to dietitian."),
    ]),

    ("allergy-and-family-medication", """New patient questionnaire, 12 Aug 2025
Allergies: penicillin (rash).
Family history: her mother takes warfarin.
Current medicines: levothyroxine 75 micrograms once daily.
""", D("12 Aug 2025", "New patient questionnaire, 12 Aug 2025"), [], [
        M("levothyroxine", "75 micrograms", "once daily", "current", "Current medicines: levothyroxine 75 micrograms once daily."),
    ], []),

    ("flag-between-value-and-unit", """RESULTS - Pine Ridge Lab (fictional) - 2025-03-03
Potassium   5.8 H  mmol/L   3.5-5.1
Sodium      133 L  mmol/L   135-145
""", D("2025-03-03", "RESULTS - Pine Ridge Lab (fictional) - 2025-03-03"), [
        L("Potassium", "5.8", "mmol/L", "3.5-5.1", None, "Potassium   5.8 H  mmol/L   3.5-5.1"),
        L("Sodium", "133", "mmol/L", "135-145", None, "Sodium      133 L  mmol/L   135-145"),
    ], [], []),

    ("decimal-comma", """Laborbefund (fictional sample), 15.04.2025
Hemoglobin 13,2 g/dL (12,0-16,0)
Leukocytes 6,1 10^3/uL (4,0-10,0)
""", D("15.04.2025", "Laborbefund (fictional sample), 15.04.2025"), [
        L("Hemoglobin", "13,2", "g/dL", "12,0-16,0", None, "Hemoglobin 13,2 g/dL (12,0-16,0)"),
        L("Leukocytes", "6,1", "10^3/uL", "4,0-10,0", None, "Leukocytes 6,1 10^3/uL (4,0-10,0)"),
    ], [], []),

    ("en-dash-range", """Glucose tolerance - Brookfield Clinic (fictional)
Reported 2025-07-08
Fasting glucose 5.4 mmol/L (ref. 3.9–5.5)
""", D("2025-07-08", "Reported 2025-07-08"), [
        L("Fasting glucose", "5.4", "mmol/L", "3.9–5.5", None, "Fasting glucose 5.4 mmol/L (ref. 3.9–5.5)"),
    ], [], []),

    ("appointment-only-no-date", """Hi Mara,
This is a reminder that your appointment is on 12 May 2026 at 10:30 with the practice nurse.
Please bring your medication list.
""", None, [], [], [
        FU("your appointment is on 12 May 2026 at 10:30 with the practice nurse", "This is a reminder that your appointment is on 12 May 2026 at 10:30 with the practice nurse."),
        FU("Please bring your medication list", "Please bring your medication list."),
    ]),

    ("medication-review-no-names", """Annual review 2025-10-02
Medication list reviewed; no changes.
Weight 70.1 kg.
""", D("2025-10-02", "Annual review 2025-10-02"), [], [], []),

    ("stopped-years-ago", """Clinic letter, 6 February 2025
In 2018 she was prescribed simvastatin 20 mg, which she stopped after a few months.
She currently takes no regular medicines.
""", D("6 February 2025", "Clinic letter, 6 February 2025"), [], [
        M("simvastatin", "20 mg", None, "not_current", "In 2018 she was prescribed simvastatin 20 mg, which she stopped after a few months."),
    ], []),
]


def main():
    out = os.path.join(ROOT, "data", "processed", "challenge.jsonl")
    rows = []
    for i, (case, text, ddate, labs, meds, fus) in enumerate(CASES, 1):
        doc = Doc()
        doc.lines = text.strip("\n").split("\n")
        doc.gold = {"document_date": ddate, "labs": labs, "medications": meds, "follow_up": fus}
        problems = check_gold(doc)
        assert not problems, (case, problems)
        rows.append({
            "doc_id": f"challenge-{i:02d}-{case}", "patient_id": f"challenge-{i:02d}", "split": "challenge", "family": "challenge",
            "held_out_family": True, "kind": "challenge", "text": doc.text, "target": doc.gold, "flags": [case],
            "challenge": True, "labels": "hand-written, checked against the extraction rules; not clinically reviewed",
            "provenance": {"source": "authored fictional fixture", "generator": "pipeline/challenge/challenge_set.py"},
        })
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} challenge documents -> {out}")


if __name__ == "__main__":
    main()
