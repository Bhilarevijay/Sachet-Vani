from app import app, db, MissingChild

with app.app_context():
    try:
        cases = MissingChild.query.all()
        print(f"Found {len(cases)} cases.")
        for case in cases:
            print(f"ID: {case.id}, ReportID: {case.report_id}, DateReported: {case.date_reported}, MissingDate: {case.missing_date}")
    except Exception as e:
        print(f"Error querying database: {e}")
        import traceback
        traceback.print_exc()
