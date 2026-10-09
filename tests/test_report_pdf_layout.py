"""Real MuPDF layout checks for PDF-only report table and hash handling."""
import pytest

fitz = pytest.importorskip('fitz', reason='PDF layout acceptance requires PyMuPDF')

from evomind_runtime import report_render as module


DIGEST = 'ea137c3af4ff' + '1' * 52
HEADER = (234/255, 241/255, 239/255)


def fixture_html():
    return """<!doctype html><html><head><style>
    body {color:#102823;background:#fff;font-family:sans-serif;font-size:10pt;line-height:1.5;}
    table {width:100%;border-collapse:collapse;font-size:8pt;} th,td {border:1px solid #cddcd8;padding:6px;}
    th {background:#eaf1ef;text-align:left;} @media print {thead {display:table-header-group;}}
    </style></head><body><h1>Synthetic report layout</h1>
    <table><thead><tr><th>Artifact</th><th>SHA-256</th></tr></thead>
    <tbody><tr><td>summary.json</td><td>""" + DIGEST + """</td></tr></tbody></table>
    <p style="page-break-before:always">Only this paragraph belongs on the last page.</p>
    </body></html>"""


def header_backgrounds(page):
    return [drawing for drawing in page.get_drawings() if drawing.get('fill')
            and all(abs(a-b)<0.001 for a,b in zip(drawing['fill'], HEADER))]


def test_finished_table_does_not_leave_header_ghosts_on_later_page(tmp_path):
    output = tmp_path/'pdf-only-layout.pdf'
    module.render_pdf(fixture_html(), output)
    with fitz.open(output) as pdf:
        assert pdf.page_count == 2
        assert header_backgrounds(pdf[0]), 'real first-page header styling should remain'
        assert 'Only this paragraph' in pdf[1].get_text()
        assert not header_backgrounds(pdf[1]), 'finished table header must not be painted on following pages'


def test_hash_glyphs_remain_exact_without_ligature_normalization(tmp_path):
    output = tmp_path/'pdf-hash-glyphs.pdf'
    module.render_pdf(fixture_html(), output)
    with fitz.open(output) as pdf:
        assert DIGEST in ''.join(page.get_text() for page in pdf)


def test_pdf_workaround_does_not_change_the_web_html(tmp_path):
    original = fixture_html()
    module.render_pdf(original, tmp_path/'separate-format.pdf')
    assert original == fixture_html()
    assert '<thead>' in original and 'table-header-group' in original
    assert 'report-pdf-hash' not in original


def test_multi_page_table_keeps_each_data_row_once(tmp_path):
    rows = ''.join('<tr><td>row-%03d</td><td>%s</td></tr>' % (index, DIGEST) for index in range(65))
    html = fixture_html().replace('<tr><td>summary.json</td><td>'+DIGEST+'</td></tr>', rows)
    output = tmp_path/'multi-page-table.pdf'
    module.render_pdf(html, output)
    with fitz.open(output) as pdf:
        text = ''.join(page.get_text() for page in pdf)
        assert pdf.page_count >= 3
        assert all(text.count('row-%03d' % index)==1 for index in range(65))
        assert text.count(DIGEST)==65
        assert not header_backgrounds(pdf[-1])
