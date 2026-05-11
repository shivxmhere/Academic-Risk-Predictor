import streamlit as st
import pandas as pd
import datetime
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def add_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 9)
    # Footer on every page
    canvas.drawString(72, 30, "Academic Risk Predictor | Group 174 | IIT Patna | Confidential")
    canvas.drawRightString(A4[0] - 72, 30, f"Page {doc.page}")
    canvas.restoreState()

def show_export_page():
    st.title("📄 Export Report")
    
    if 'results' not in st.session_state or st.session_state['results'] is None:
        st.warning("No prediction results found. Please go to the 'Predict Risk' page and run predictions first.")
        return
        
    df = st.session_state['results']
    
    st.subheader("Report Preview Metrics")
    total_students = len(df)
    high_risk_count = (df['Predicted Risk'] == 'High Risk').sum()
    low_risk_count = (df['Predicted Risk'] == 'Low Risk').sum()
    medium_risk_count = (df['Predicted Risk'] == 'Medium Risk').sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", total_students)
    col2.metric("High Risk Count", high_risk_count)
    col3.metric("Low Risk Count", low_risk_count)
    
    st.markdown("---")
    
    if st.button("Generate PDF Report"):
        try:
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=50)
            elements = []
            
            styles = getSampleStyleSheet()
            title_style = styles['Title']
            h2_style = styles['Heading2']
            normal_style = styles['Normal']
            
            # Header
            elements.append(Paragraph("Academic Risk Predictor — Class Risk Report", title_style))
            elements.append(Paragraph(f"Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
            elements.append(Paragraph("Group 174 | IIT Patna | CSDA 2025-26", normal_style))
            elements.append(Spacer(1, 20))
            
            # Summary Section
            elements.append(Paragraph("Summary", h2_style))
            
            high_pct = (high_risk_count / total_students * 100) if total_students else 0
            med_pct = (medium_risk_count / total_students * 100) if total_students else 0
            low_pct = (low_risk_count / total_students * 100) if total_students else 0
            
            summary_data = [
                ["Metric", "Value"],
                ["Total Students", str(total_students)],
                ["Low Risk", f"{low_risk_count} ({low_pct:.1f}%)"],
                ["Medium Risk", f"{medium_risk_count} ({med_pct:.1f}%)"],
                ["High Risk", f"{high_risk_count} ({high_pct:.1f}%)"],
                ["Model Used", "Random Forest"],
                ["Model Accuracy", "91.4%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[200, 200])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#000080')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 12),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # High Risk Students List
            elements.append(Paragraph("High Risk Students", h2_style))
            high_risk_df = df[df['Predicted Risk'] == 'High Risk']
            if len(high_risk_df) > 0:
                hr_data = [["Student ID", "Attendance %", "Action"]]
                for _, row in high_risk_df.iterrows():
                    sid = row.get('student_id', 'Unknown')
                    att = f"{row['attendance_pct']}%"
                    action = "Immediately contact the student, alert their faculty advisor, and schedule a counselling session."
                    
                    # Wrap action text so it fits in the table column without overflowing
                    action_para = Paragraph(action, normal_style)
                    hr_data.append([str(sid), att, action_para])
                    
                hr_table = Table(hr_data, colWidths=[80, 80, 340])
                hr_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.darkred),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ]))
                elements.append(hr_table)
            else:
                elements.append(Paragraph("No High Risk students detected.", normal_style))
            
            elements.append(PageBreak())
            
            # Full Results Table
            elements.append(Paragraph("Full Risk Results", h2_style))
            
            table_data = [["Student ID", "Attend %", "Int Score", "Asgn Marks", "Predicted Risk", "Conf %"]]
            for _, row in df.iterrows():
                sid = str(row.get('student_id', 'N/A'))
                att = f"{row['attendance_pct']}"
                int_sc = f"{row['internal_score']}"
                asg = f"{row['assignment_marks']}"
                risk = str(row['Predicted Risk'])
                conf = f"{row['Confidence %']}%"
                
                # Format Risk string with colors for ReportLab using Paragraph with inline XML styling
                if risk == 'High Risk':
                    risk_para = Paragraph(f"<font color='red'><b>{risk}</b></font>", normal_style)
                elif risk == 'Medium Risk':
                    risk_para = Paragraph(f"<font color='orange'><b>{risk}</b></font>", normal_style)
                else:
                    risk_para = Paragraph(f"<font color='green'><b>{risk}</b></font>", normal_style)
                
                table_data.append([sid, att, int_sc, asg, risk_para, conf])
                
            full_table = Table(table_data, colWidths=[80, 60, 60, 70, 120, 60], repeatRows=1)
            full_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#000080')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            
            elements.append(full_table)
            
            # Build PDF
            doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
            
            st.success("PDF generated successfully!")
            
            st.download_button(
                label="⬇ Download PDF Report", 
                data=buffer.getvalue(), 
                file_name="ARP_Risk_Report_Group174.pdf", 
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"An error occurred while generating the PDF: {str(e)}")
