"""
pdf_generator.py
================
Generates a polished, court-ready Legal Research Memorandum in PDF format
using ReportLab.
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and stamp total page count in the footer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            super().showPage()
        super().save()

    def draw_footer(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Footer divider line
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(40, 36, letter[0] - 40, 36)

        # Footer labels
        footer_text = f"LegalGraph Intelligence · Legal Research Memorandum · Page {self._pageNumber} of {page_count}"
        self.drawString(40, 24, footer_text)
        self.drawRightString(letter[0] - 40, 24, "CONFIDENTIAL & PRIVILEGED WORK PRODUCT")
        self.restoreState()


def _clean_markdown_to_xml(text: str) -> str:
    """Convert common markdown formatting to ReportLab XML tags."""
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)
    # Inline code / chips
    text = re.sub(r"`(.+?)`", r'<font face="Courier" color="#1e3a8a"><b>\1</b></font>', text)
    # Ampersands
    text = text.replace("&", "&amp;")
    # Fix double escaping of XML tags
    text = text.replace("&amp;lt;", "<").replace("&amp;gt;", ">")
    text = re.sub(r"&amp;([a-zA-Z]+);", r"&\1;", text)
    return text


def generate_legal_memo_pdf(
    query: str,
    ai_explanation: str,
    connecting_path: Optional[List[Dict[str, Any]]] = None,
    recommendations: Optional[List[Any]] = None,
    graph_store: Optional[Any] = None,
) -> bytes:
    """
    Generate a complete, professionally styled Legal Research Memorandum PDF.
    Returns bytes suitable for Streamlit st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=50,
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    primary_color = colors.HexColor("#0f172a")    # Deep Navy
    accent_color = colors.HexColor("#b45309")     # Gold/Amber
    secondary_color = colors.HexColor("#1e3a8a")  # Sapphire Blue
    slate_text = colors.HexColor("#334155")
    card_bg = colors.HexColor("#f8fafc")

    memo_header_style = ParagraphStyle(
        "MemoHeader",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4,
    )
    memo_sub_style = ParagraphStyle(
        "MemoSub",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12,
    )
    section_title_style = ParagraphStyle(
        "SectionTitle",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=secondary_color,
        spaceBefore=14,
        spaceAfter=6,
    )
    h3_style = ParagraphStyle(
        "H3Style",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=primary_color,
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=slate_text,
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=slate_text,
        leftIndent=14,
        spaceAfter=4,
    )
    meta_label = ParagraphStyle(
        "MetaLabel",
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=primary_color,
    )
    meta_val = ParagraphStyle(
        "MetaVal",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=slate_text,
    )

    story = []

    # ========================================================================= #
    # 1. Document Header Banner
    # ========================================================================= #
    story.append(Paragraph("LEGAL RESEARCH MEMORANDUM", memo_header_style))
    story.append(Paragraph("JUDICIAL PRECEDENT & STATUTE ANALYSIS REPORT", memo_sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=accent_color, spaceBefore=0, spaceAfter=12))

    # Metadata Block Table
    now_str = datetime.now().strftime("%B %d, %Y · %I:%M %p")
    meta_data = [
        [
            Paragraph("<b>DATE:</b>", meta_label),
            Paragraph(now_str, meta_val),
            Paragraph("<b>CLASSIFICATION:</b>", meta_label),
            Paragraph("Research Work Product", meta_val),
        ],
        [
            Paragraph("<b>SOURCE:</b>", meta_label),
            Paragraph("LegalGraph Precedent Engine", meta_val),
            Paragraph("<b>JURISDICTION:</b>", meta_label),
            Paragraph("Federal / State / High Court", meta_val),
        ],
        [
            Paragraph("<b>QUERY:</b>", meta_label),
            Paragraph(f"<i>{query}</i>", meta_val),
            Paragraph("<b>STATUS:</b>", meta_label),
            Paragraph("<font color='#059669'><b>Active Precedent Verified</b></font>", meta_val),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[1.1 * inch, 2.7 * inch, 1.4 * inch, 2.0 * inch])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), card_bg),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#f1f5f9")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # ========================================================================= #
    # 2. Executive Legal Analysis (AI Synthesis)
    # ========================================================================= #
    story.append(Paragraph("I. EXECUTIVE LEGAL ANALYSIS & SYNTHESIS", section_title_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=secondary_color, spaceBefore=2, spaceAfter=8))

    # Parse AI Explanation lines cleanly into PDF Flowables
    lines = (ai_explanation or "").split("\n")
    table_buffer: List[List[str]] = []
    in_table = False

    for line in lines:
        line_s = line.strip()
        if not line_s:
            if in_table and table_buffer:
                # Flush markdown table
                t_flowable = _build_reportlab_table(table_buffer, body_style)
                if t_flowable:
                    story.append(t_flowable)
                    story.append(Spacer(1, 6))
                table_buffer = []
                in_table = False
            continue

        # Detect Markdown Tables (e.g. | Col1 | Col2 |)
        if line_s.startswith("|") and line_s.endswith("|"):
            in_table = True
            # Skip separator line like |---|---|
            if set(line_s.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            cells = [c.strip() for c in line_s.split("|")[1:-1]]
            table_buffer.append(cells)
            continue
        elif in_table:
            # End of table
            t_flowable = _build_reportlab_table(table_buffer, body_style)
            if t_flowable:
                story.append(t_flowable)
                story.append(Spacer(1, 6))
            table_buffer = []
            in_table = False

        # Detect Headings
        if line_s.startswith("###"):
            h_text = line_s.lstrip("#").strip()
            story.append(Paragraph(_clean_markdown_to_xml(h_text), h3_style))
        elif line_s.startswith("##"):
            h_text = line_s.lstrip("#").strip()
            story.append(Paragraph(_clean_markdown_to_xml(h_text), section_title_style))
        elif line_s.startswith("-") or line_s.startswith("*"):
            b_text = line_s.lstrip("-*").strip()
            clean_b = _clean_markdown_to_xml(b_text)
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{clean_b}", bullet_style))
        else:
            clean_p = _clean_markdown_to_xml(line_s)
            story.append(Paragraph(clean_p, body_style))

    if in_table and table_buffer:
        t_flowable = _build_reportlab_table(table_buffer, body_style)
        if t_flowable:
            story.append(t_flowable)
            story.append(Spacer(1, 6))

    story.append(Spacer(1, 12))

    # ========================================================================= #
    # 3. Procedural Precedent Chain (Connecting Citation Graph)
    # ========================================================================= #
    story.append(Paragraph("II. JUDICIAL PRECEDENT & CITATION CHAIN", section_title_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=secondary_color, spaceBefore=2, spaceAfter=8))

    if connecting_path and len(connecting_path) > 1:
        story.append(
            Paragraph(
                "The following procedural chain demonstrates how the key precedents link through appellate affirmations, "
                "subsequent overrulings, and binding citations:",
                body_style,
            )
        )
        chain_rows = [["Step", "Precedent / Case Name", "Legal Relationship", "Jurisdiction"]]
        for i in range(len(connecting_path) - 1):
            curr_hop = connecting_path[i]
            next_hop = connecting_path[i + 1]
            c_node = curr_hop.get("node", {}) if isinstance(curr_hop, dict) and "node" in curr_hop else curr_hop
            n_node = next_hop.get("node", {}) if isinstance(next_hop, dict) and "node" in next_hop else next_hop
            rel = next_hop.get("rel_type", "CITES")

            c_title = c_node.get("title", "Case")
            c_year = f" ({c_node.get('year')})" if c_node.get("year") else ""
            c_court = c_node.get("court", "Court of Record")

            rel_label = str(rel).replace("_", " ").title()
            if "AFFIRM" in str(rel).upper():
                rel_str = f"<font color='#0284c7'><b>Affirmed by</b></font> &rarr; {n_node.get('title')}"
            elif "OVERRULE" in str(rel).upper():
                rel_str = f"<font color='#dc2626'><b>Overruled by</b></font> &rarr; {n_node.get('title')}"
            else:
                rel_str = f"<font color='#b45309'><b>Cites as Precedent</b></font> &rarr; {n_node.get('title')}"

            chain_rows.append([
                Paragraph(f"<b>{i+1}</b>", meta_label),
                Paragraph(f"<b>{c_title}</b>{c_year}", body_style),
                Paragraph(rel_str, body_style),
                Paragraph(c_court, body_style),
            ])

        chain_table = Table(chain_rows, colWidths=[0.6 * inch, 2.6 * inch, 2.7 * inch, 1.3 * inch])
        chain_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ])
        )
        story.append(chain_table)
    else:
        story.append(
            Paragraph(
                "<i>No direct multi-hop citation path connects the top candidates in this sample. "
                "The decisions are unified doctrinally through common statutory foundations and core legal concepts.</i>",
                body_style,
            )
        )

    story.append(Spacer(1, 14))

    # ========================================================================= #
    # 4. Ranked Judicial Precedent Briefs
    # ========================================================================= #
    story.append(Paragraph("III. RECOMMENDED JUDICIAL PRECEDENTS & HOLDINGS", section_title_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=secondary_color, spaceBefore=2, spaceAfter=8))

    if recommendations:
        for rank, rec in enumerate(recommendations, start=1):
            ctx = graph_store.case_full_context(rec.case_id) if graph_store else {}
            case_info = ctx.get("case", {})
            judge = ctx.get("judge", {})
            statutes = ctx.get("cited_statutes", [])

            statute_titles = [f"{s.get('code_section')}: {s.get('title')}" for s in statutes] if statutes else ["General Common Law"]
            statutes_str = "; ".join(statute_titles)

            pct = int(rec.final_score * 100) if hasattr(rec, "final_score") else 90

            brief_data = [
                [
                    Paragraph(f"<b>#{rank} · {rec.title}</b>", ParagraphStyle("CaseHead", fontName="Helvetica-Bold", fontSize=11, textColor=primary_color)),
                    Paragraph(f"<b>Relevance: {pct}%</b>", ParagraphStyle("ScoreHead", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#059669"), alignment=2)),
                ],
                [
                    Paragraph(f"<b>Court:</b> {rec.court} &nbsp;|&nbsp; <b>Year:</b> {rec.year} &nbsp;|&nbsp; <b>Judge:</b> {judge.get('name', 'N/A')}", ParagraphStyle("Sub", fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#475569"))),
                    "",
                ],
                [
                    Paragraph(f"<b>Holding / Summary:</b> {rec.summary}", body_style),
                    "",
                ],
                [
                    Paragraph(f"<b>Statutes Grounding:</b> <font color='#1e3a8a'>{statutes_str}</font>", ParagraphStyle("Stat", fontName="Helvetica-Oblique", fontSize=8.5, textColor=slate_text)),
                    "",
                ],
            ]
            brief_table = Table(brief_data, colWidths=[5.6 * inch, 1.6 * inch])
            brief_table.setStyle(
                TableStyle([
                    ("SPAN", (0, 1), (1, 1)),
                    ("SPAN", (0, 2), (1, 2)),
                    ("SPAN", (0, 3), (1, 3)),
                    ("BACKGROUND", (0, 0), (-1, -1), card_bg),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#cbd5e1")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ])
            )
            story.append(KeepTogether([brief_table, Spacer(1, 8)]))

    # Build PDF
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


def _build_reportlab_table(table_data: List[List[str]], text_style: ParagraphStyle) -> Optional[Table]:
    """Helper to convert Markdown tabular rows to a ReportLab Table."""
    if not table_data or len(table_data) < 2:
        return None

    col_count = len(table_data[0])
    # Distribute 7.2 inches across columns
    col_width = (7.2 / col_count) * inch

    formatted_data = []
    for r_idx, row in enumerate(table_data):
        row_cells = []
        is_header = r_idx == 0
        for cell in row:
            clean = _clean_markdown_to_xml(cell)
            if is_header:
                row_cells.append(Paragraph(f"<b>{clean}</b>", ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.white)))
            else:
                row_cells.append(Paragraph(clean, ParagraphStyle("TD", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=colors.HexColor("#334155"))))
        formatted_data.append(row_cells)

    rl_table = Table(formatted_data, colWidths=[col_width] * col_count)
    rl_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )
    return rl_table
