"""Render the final one-page research decision and claim/evidence map."""
from pathlib import Path
import json

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ROOT=Path(__file__).resolve().parents[1]
FONT='/usr/share/fonts/truetype/dejavu/'
pdfmetrics.registerFont(TTFont('DV',FONT+'DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DV-Bold',FONT+'DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
NAVY=colors.HexColor('#183142')
TEAL=colors.HexColor('#13676c')
LIGHT=colors.HexColor('#edf3f4')
GRAY=colors.HexColor('#5d6970')


def style(name,size=10,leading=13.1,**kwargs):
    options=dict(fontName='DV',fontSize=size,leading=leading,textColor=NAVY,spaceAfter=6)
    options.update(kwargs)
    return ParagraphStyle(name,**options)


BODY=style('body')
SMALL=style('small',8.2,10.7,textColor=GRAY)
HEAD=style('head',11,14,fontName='DV-Bold',textColor=TEAL,spaceBefore=7)
TITLE=style('title',21,25,fontName='DV-Bold',spaceAfter=4)


def p(text,sty=BODY):
    return Paragraph(text,sty)


def table(rows,widths,size=8.9):
    st=style('cell',size,size*1.24,spaceAfter=0)
    cells=[[p(str(x),st) for x in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),LIGHT),
        ('LINEBELOW',(0,0),(-1,0),.7,TEAL),
        ('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#c4d0d3')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    return t


def footer(canvas,doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#c4d0d3'))
    canvas.line(doc.leftMargin,32,doc.pagesize[0]-doc.rightMargin,32)
    canvas.setFont('DV',7.3);canvas.setFillColor(GRAY)
    canvas.drawString(doc.leftMargin,21,'Frozen R15 preserved • Protocol date 24 Nov 2026 • Evidence and limits in the research package')
    canvas.restoreState()


def decision():
    f=ROOT/'reports/RESEARCH_DECISION.pdf'
    doc=SimpleDocTemplate(str(f),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=42,
        title='J-CERT+ bounded VLDB research decision',author='J-CERT+ research workstream')
    story=[p('J-CERT+ | Research decision',TITLE),
      p('Evidence first. No manuscript promotion yet.',style('subtitle',11,14,textColor=TEAL)),
      p('<b>Decision.</b> Retain the full-guarantee cost result; do not promote the current adaptive anchor controller. '
        'Its outputs remain certified in the checked scope, but the fixed held-out experiment fails the cost-benefit gate. '
        'Stop joint-scale expansion and retuning on this confirmation seed.'),
      p('1  What full-catalog coverage costs',HEAD),
      p('With equal root-factor/catalog reuse, the preselected E1 seed gave identical free shortlist and D trajectories. '
        'The shortlist still paid full-catalog caps and global top-64 selection. D paid the following additional complete Q4 cost:'),
      table([['Paired mode','D / shortlist','D − shortlist, seconds'],
             ['Unit; all six configurations','1.035–1.130','+0.028 to +0.185'],
             ['Natural weights; all six','1.381–1.496','+0.340 to +0.840']], [173,144,198]),
      Spacer(1,5),
      p('2  The unchanged controller fails on held-out streams',HEAD)]
    groups=json.loads((ROOT/'analysis/a1_confirmation/summary.json').read_text())['groups']
    rows=[['Graph / opinion stream','Rebuild / retain','Rebuild / adaptive','Adaptive − rebuild (s)']]
    for graph in ('Alpha','OTC'):
        for family in ('slow','independent','signed-neighbor'):
            arms={x['arm']:x for x in groups if x['graph']==graph and x['stream']==family}
            ret=arms['retain-always']['prefixes']['16']['rebuild_over_arm']['median']
            ad=arms['adaptive']['prefixes']['16']['rebuild_over_arm']['median']
            delta=arms['adaptive']['prefixes']['16']['arm_minus_rebuild_s']['median']
            rows.append([graph+' / '+family,f'{ret:.3f}',f'{ad:.3f}',f'{delta:+.3f}'])
    story.extend([table(rows,[182,103,108,122],8.5),Spacer(1,4),
      p('Ratios above one favor reuse. Q16 includes common structural preparation and all reset/opinion/online work. '
        'Each cell is the median of three paired repeats on one frozen input, not a population confidence interval.',SMALL),
      p('The policies solved the same response workload. Retention expanded endpoint checks; a first-cold-frontier '
        'forecast underestimated later child versions. Slow Alpha’s adaptive advantage shrank from 1.191 at Q8 to '
        '1.035 at Q16. Cheaper valid index construction could shift this implementation-specific crossover.'),
      p('3  What is established—and what stays bounded',HEAD),
      p('<b>Established:</b> paired natural-weight support; fair persistent-root lifecycle and exclusive cost ledger; '
        'F/T/I/D/S measurements; representative independent full-catalog proofs; common-kernel P/L attribution. '
        'The held-out 54 workers completed 864 requests with 12,960 matching winners and gain intervals. '
        'Independent replay separately proved the 195 decisions in 13 preselected requests.'),
      p('<b>Bounded:</b> A2 is a fixed-version, equal-weight all-nonedges microprototype. Positive-tolerance '
        'screening has its own tested API; the large service stays at δ=0. No precision-scheduling escalation. '
        'Fresh shadow replay stopped at its 180 s cap: 330 matching steps, six requests uncovered.'),
      p('<b>Next research question:</b> predict the future access cost of retained evidence, rather than assume '
        'that a currently selective anchor stays cheap. Use a new development/confirmation split for a new policy.'),
      p('Format probe: official fonts/template, 14 pages; body plus AI statement ends on page 13, references on 14. '
        'All body floats end on 12. It is a capacity probe, not a submission-ready paper.',SMALL)])
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def claims():
    f=ROOT/'reports/CLAIM_EVIDENCE_MAP.pdf'
    doc=SimpleDocTemplate(str(f),pagesize=landscape(A4),leftMargin=35,rightMargin=35,topMargin=30,bottomMargin=42,
        title='J-CERT+ claim and evidence map',author='J-CERT+ research workstream')
    rows=[['Claim / responsibility','Frozen or new evidence','Boundary that remains'],
      ['A shortlist can miss later contenders',
       'R15 controlled cases stay unchanged. New pilot: two insertion identities outside top-64 at six positions in three adjacent OTC weighted/slow requests.',
       'All 1,260 audited winning positions had positive root gain. Outsiders are not negative-root revival events or independent frequencies.'],
      ['Complete coverage has a measurable cost',
       'E1: 180 completed F/T/I/D/S workers, identical free S paths on the fixed seed; Q4 paired D/S costs and absolute differences are saved.',
       'S pays full-catalog preparation/caps and global top-64 selection (partition, not a full sort). Repeats and query prefixes are correlated.'],
      ['Root preparation can legally persist',
       '216 F/D pilot workers on paired unit/natural-weight projections; same immutable root factor and catalog privileges, charged resets, empty request Grams.',
       'Requests restart at a fixed root. This is not an arbitrary external graph-update service or free rollback.'],
      ['Tangent evidence can cross opinion changes',
       'Bounded-state proof, source-bound weighted adapter, index-event admission and actual timed output replay; all A1 arms preserve current decisions.',
       'The unchanged adaptive policy fails the Q16 cost gate. The mathematical envelope is valid even when it is expensive.'],
      ['Generic numerical speed and maintenance differ',
       '42 common-kernel P/L plus original-Q cells match independently checked reference paths. Generic fused kernels benefit both P and L.',
       'P loses in some shared kernels; results do not prove universal superiority or pure routing attribution.'],
      ['Certificates validate decisions',
       'Current operator/catalog reconstruction and integer-residual energy bounds. Held-out sample:195 verified decisions,180 distinct per-input steps.',
       'Independent sample replay differs from all-run agreement. Fresh shadow:22/28 requests complete; six uncovered at its cap.'],
      ['A2 and precision theory have defined scope',
       'A2 exact small-domain checks and equal-weight monotone heap proof. Prior ε threshold and positive-δ acceptance boundaries audited.',
       'No two-hop implicit-enumeration scaling, no new refresh theorem, no positive-δ large-service speed claim or trajectory-probability claim.']]
    story=[p('J-CERT+ | Claim–evidence map',TITLE),p('One frozen baseline; separate new research evidence and explicit decision gates.',BODY),
      table(rows,[185,302,285],8.5),Spacer(1,8),
      p('Detailed records: E1_GUARANTEE_COST.txt · A1_CONFIRMATION_RESULTS.txt · VERIFICATION_COVERAGE.md · '
        'MATH_PROTOTYPES.txt · PARITY_AND_PROBABILISTIC_INTERFACES.txt. R15’s 1,581 baseline files are unchanged.',SMALL)]
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=='__main__':
    decision();claims()
