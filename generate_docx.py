import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding in twips (1/20th of a point)."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tc_pr.append(tc_mar)

def set_table_borders(table, color="CCCCCC", sz="4", val="single"):
    """Applies subtle borders to a table."""
    tbl_pr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl_pr.append(borders)

def build_case_study_docx():
    md_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CASE_STUDY.md")
    output_docx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HealthPulse_DWM_Case_Study.docx")
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    doc = Document()

    # Page Margins: 1 inch on all sides
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
        # Header & Footer setup
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("HealthPulse: Clinical Data Warehouse & Decision Support System")
        hrun.font.name = "Calibri"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(140, 150, 160)

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("Apex Metropolitan Health System — Confidential Case Study")
        frun.font.name = "Calibri"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(140, 150, 160)

    # Base Styles
    style_normal = doc.styles['Normal']
    style_normal.font.name = 'Calibri'
    style_normal.font.size = Pt(11)
    style_normal.font.color.rgb = RGBColor(30, 41, 59) # Slate 800
    style_normal.paragraph_format.line_spacing = 1.18
    style_normal.paragraph_format.space_after = Pt(6)

    # ---------------------------------------------------------
    # COVER / TITLE PAGE
    # ---------------------------------------------------------
    # Top spacing
    p_top = doc.add_paragraph()
    p_top.paragraph_format.space_before = Pt(80)

    # Super-title badge
    p_badge = doc.add_paragraph()
    r_badge = p_badge.add_run("ENTERPRISE TECHNICAL CASE STUDY & PROJECT REPORT")
    r_badge.font.name = "Calibri"
    r_badge.font.size = Pt(10.5)
    r_badge.font.bold = True
    r_badge.font.color.rgb = RGBColor(14, 116, 144) # Teal 700

    # Main Title
    p_title = doc.add_paragraph()
    r_title = p_title.add_run("HealthPulse: Clinical Data Warehouse & Predictive Healthcare Decision Support System")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(26)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42) # Slate 900
    p_title.paragraph_format.space_after = Pt(12)

    # Subtitle
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run("A Unified Multi-Dimensional Star Schema Warehouse, Supervised Diagnostic Mining, and Comorbidity Analytics Framework for Tertiary Hospital Systems")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(13)
    r_sub.font.color.rgb = RGBColor(71, 85, 105) # Slate 600
    p_sub.paragraph_format.space_after = Pt(180)

    # Metadata Box
    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_before = Pt(40)
    p_meta.paragraph_format.line_spacing = 1.3
    
    runs_meta = [
        ("Specialization / Subject: ", True),
        ("Data Warehousing and Data Mining (DWM) / Healthcare Informatics\n", False),
        ("Target Organization: ", True),
        ("Apex Metropolitan Health System (Regional Tertiary Medical Center)\n", False),
        ("System Architecture: ", True),
        ("Star Schema Dimensional Warehouse (SQLite) + Multi-Model Machine Learning\n", False),
        ("Implementation Stack: ", True),
        ("Python 3.13, Streamlit, Scikit-Learn, Plotly, MLxtend, Pandas, NumPy\n", False),
        ("Document Classification: ", True),
        ("Academic Capstone & Technical Architecture Project Report\n", False),
        ("Date / Version: ", True),
        ("Academic Year 2025-2026 | Version 2.4 Production Release", False)
    ]
    for text, bold in runs_meta:
        r = p_meta.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10)
        r.font.bold = bold
        if bold:
            r.font.color.rgb = RGBColor(14, 116, 144)
        else:
            r.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_page_break()

    # ---------------------------------------------------------
    # PARSE MARKDOWN INTO NATIVE WORD ELEMENTS
    # ---------------------------------------------------------
    lines = md_text.split("\n")
    i = 0
    in_code_block = False
    code_block_lines = []
    
    in_table = False
    table_lines = []

    def flush_table(tbl_lines):
        if not tbl_lines:
            return
        
        # Clean lines
        rows_data = []
        for l in tbl_lines:
            if re.match(r'^\s*\|?\s*[-:]+[-| :]*\|?\s*$', l):
                continue # Skip separator line
            cols = [c.strip() for c in l.strip().strip("|").split("|")]
            if cols:
                rows_data.append(cols)
                
        if not rows_data:
            return
            
        n_rows = len(rows_data)
        n_cols = max(len(r) for r in rows_data)
        
        table = doc.add_table(rows=n_rows, cols=n_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True
        set_table_borders(table, color="CBD5E1", sz="4")

        for r_idx, row in enumerate(rows_data):
            is_header = (r_idx == 0)
            for c_idx in range(n_cols):
                cell = table.cell(r_idx, c_idx)
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_margins(cell, top=120, bottom=120, left=160, right=160)
                
                text_val = row[c_idx] if c_idx < len(row) else ""
                # Strip markdown bolding or math
                clean_text = text_val.replace("**", "").replace("$", "")
                
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.05
                
                run = p.add_run(clean_text)
                run.font.name = "Calibri"
                
                if is_header:
                    set_cell_background(cell, "0E7490") # Teal header
                    run.font.bold = True
                    run.font.size = Pt(9.5)
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                else:
                    if r_idx % 2 == 1:
                        set_cell_background(cell, "F8FAFC") # Light slate zebra
                    else:
                        set_cell_background(cell, "FFFFFF")
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(30, 41, 59)
                    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        p_spacer = doc.add_paragraph()
        p_spacer.paragraph_format.space_after = Pt(4)

    def flush_code_block(c_lines):
        if not c_lines:
            return
        
        # Create shaded callout table for code
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        set_cell_background(cell, "F1F5F9") # Slate 100
        set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
        set_table_borders(tbl, color="CBD5E1", sz="6")
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        
        code_str = "\n".join(c_lines)
        run = p.add_run(code_str)
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
        run.font.color.rgb = RGBColor(15, 23, 42)
        
        p_spacer = doc.add_paragraph()
        p_spacer.paragraph_format.space_after = Pt(4)

    def add_styled_paragraph(text, bullet=False, quote=False):
        if not text.strip():
            return
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.18
        p.paragraph_format.space_after = Pt(6)
        
        if bullet:
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_after = Pt(3)
            # Add bullet symbol
            r_bullet = p.add_run("•  ")
            r_bullet.font.name = "Calibri"
            r_bullet.font.bold = True
            r_bullet.font.color.rgb = RGBColor(14, 116, 144)

        if quote:
            p.paragraph_format.left_indent = Inches(0.4)
            p.paragraph_format.right_indent = Inches(0.2)

        # Parse inline tokens: **bold**, *italic*, `code`, math $...$
        tokens = re.split(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\$.*?\$)', text)
        for tok in tokens:
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                r = p.add_run(tok[2:-2])
                r.font.bold = True
                r.font.color.rgb = RGBColor(15, 23, 42)
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 1:
                r = p.add_run(tok[1:-1])
                r.font.italic = True
                r.font.color.rgb = RGBColor(51, 65, 85)
            elif tok.startswith("`") and tok.endswith("`"):
                r = p.add_run(tok[1:-1])
                r.font.name = "Consolas"
                r.font.size = Pt(9.5)
                r.font.color.rgb = RGBColor(15, 118, 110)
            elif tok.startswith("$") and tok.endswith("$"):
                # Mathematical notation
                r = p.add_run(tok.replace("$", ""))
                r.font.italic = True
                r.font.color.rgb = RGBColor(30, 41, 59)
            else:
                r = p.add_run(tok)
                r.font.name = "Calibri"
                r.font.color.rgb = RGBColor(30, 41, 59)

    # Process all lines
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle Code blocks
        if stripped.startswith("```"):
            if in_code_block:
                flush_code_block(code_block_lines)
                code_block_lines = []
                in_code_block = False
            else:
                if in_table:
                    flush_table(table_lines)
                    table_lines = []
                    in_table = False
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue

        # Handle Tables
        if "|" in stripped and (stripped.startswith("|") or stripped.endswith("|")):
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(stripped)
            i += 1
            continue
        else:
            if in_table:
                flush_table(table_lines)
                table_lines = []
                in_table = False

        # Ignore mermaid diagrams (since Word doesn't natively render them, code block handles it)
        if stripped.startswith("```mermaid"):
            # skip until end of code block
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            # Heading 1 (e.g. Chapter)
            h_text = stripped[2:].strip()
            if "CASE STUDY:" in h_text:
                i += 1
                continue # Already in title page
            p_h = doc.add_heading(level=1)
            p_h.paragraph_format.space_before = Pt(22)
            p_h.paragraph_format.space_after = Pt(8)
            p_h.paragraph_format.keep_with_next = True
            run_h = p_h.add_run(h_text)
            run_h.font.name = "Calibri"
            run_h.font.size = Pt(17)
            run_h.font.bold = True
            run_h.font.color.rgb = RGBColor(14, 116, 144) # Teal 700

        elif stripped.startswith("## "):
            # Heading 2 (e.g. Chapter title or Executive Summary)
            h_text = stripped[3:].strip()
            
            # Start major chapters on a fresh page
            if h_text.startswith("Chapter") or "Table of Contents" in h_text or "References" in h_text:
                doc.add_page_break()
                
            p_h = doc.add_heading(level=2)
            p_h.paragraph_format.space_before = Pt(14)
            p_h.paragraph_format.space_after = Pt(6)
            p_h.paragraph_format.keep_with_next = True
            run_h = p_h.add_run(h_text)
            run_h.font.name = "Calibri"
            run_h.font.size = Pt(14)
            run_h.font.bold = True
            run_h.font.color.rgb = RGBColor(30, 58, 138) # Blue 900

        elif stripped.startswith("### "):
            # Heading 3 (Sub-chapter)
            h_text = stripped[4:].strip()
            p_h = doc.add_heading(level=3)
            p_h.paragraph_format.space_before = Pt(14)
            p_h.paragraph_format.space_after = Pt(4)
            p_h.paragraph_format.keep_with_next = True
            run_h = p_h.add_run(h_text)
            run_h.font.name = "Calibri"
            run_h.font.size = Pt(12)
            run_h.font.bold = True
            run_h.font.color.rgb = RGBColor(15, 23, 42) # Slate 900

        elif stripped.startswith("#### "):
            # Heading 4
            h_text = stripped[5:].strip()
            p_h = doc.add_heading(level=4)
            p_h.paragraph_format.space_before = Pt(10)
            p_h.paragraph_format.space_after = Pt(2)
            p_h.paragraph_format.keep_with_next = True
            run_h = p_h.add_run(h_text)
            run_h.font.name = "Calibri"
            run_h.font.size = Pt(11)
            run_h.font.bold = True
            run_h.font.color.rgb = RGBColor(71, 85, 105)

        elif stripped.startswith("- ") or stripped.startswith("* "):
            add_styled_paragraph(stripped[2:], bullet=True)

        elif re.match(r'^\d+\.\s', stripped):
            add_styled_paragraph(stripped, bullet=False)

        elif stripped.startswith("> "):
            add_styled_paragraph(stripped[2:], quote=True)

        elif stripped.startswith("---"):
            # Subtle section divider or page break if between major chapters
            p_div = doc.add_paragraph()
            p_div.paragraph_format.space_before = Pt(8)
            p_div.paragraph_format.space_after = Pt(8)

        else:
            if stripped:
                add_styled_paragraph(stripped)

        i += 1

    if in_table:
        flush_table(table_lines)
    if in_code_block:
        flush_code_block(code_block_lines)

    doc.save(output_docx)
    print(f"Case study Word document saved successfully: {output_docx}")
    file_size_kb = round(os.path.getsize(output_docx) / 1024, 2)
    print(f"File size: {file_size_kb} KB")
    return output_docx

if __name__ == "__main__":
    build_case_study_docx()
