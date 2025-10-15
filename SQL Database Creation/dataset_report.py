import pandas as pd
from sqlalchemy import create_engine, inspect
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# === CONFIG ===
DATABASE_URL = "postgresql+psycopg2://postgres:5737@localhost:5433/intelliscript2"
OUTPUT_FILE = "database_report.pdf"
ROW_LIMIT = 10   # number of sample rows per table

# === SETUP ===
engine = create_engine(DATABASE_URL)
inspector = inspect(engine)
styles = getSampleStyleSheet()

doc = SimpleDocTemplate(OUTPUT_FILE, pagesize=A4)
elements = []

# === LOOP THROUGH TABLES ===
for table_name in inspector.get_table_names(schema="public"):
    elements.append(Paragraph(f"Table: {table_name}", styles["Heading1"]))

    # --- Schema Info ---
    columns = inspector.get_columns(table_name, schema="public")
    schema_data = [["Column", "Type", "Nullable", "Default"]]
    for col in columns:
        schema_data.append([
            col["name"],
            str(col["type"]),
            "YES" if col["nullable"] else "NO",
            str(col["default"])
        ])
    schema_table = Table(schema_data, hAlign="LEFT")
    schema_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.black)
    ]))
    elements.append(Paragraph("Schema:", styles["Heading2"]))
    elements.append(schema_table)
    elements.append(Spacer(1, 12))

    # --- Sample Data ---
    try:
        df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
        if not df.empty:
            data = [df.columns.tolist()] + df.astype(str).values.tolist()
            data_table = Table(data, hAlign="LEFT")
            data_table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.lightblue),
                ("GRID", (0,0), (-1,-1), 0.25, colors.black),
                ("ALIGN", (0,0), (-1,-1), "LEFT")
            ]))
            elements.append(Paragraph("Sample Data:", styles["Heading2"]))
            elements.append(data_table)
            elements.append(Spacer(1, 24))
    except Exception as e:
        elements.append(Paragraph(f"Could not fetch data: {e}", styles["Normal"]))

    elements.append(PageBreak())

# === BUILD PDF ===
doc.build(elements)
print(f"Report generated: {OUTPUT_FILE}")
