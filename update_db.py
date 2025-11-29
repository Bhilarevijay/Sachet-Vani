import sqlite3

def update_db():
    conn = sqlite3.connect('missing_children.db')
    cursor = conn.cursor()
    
    columns = [
        ('abduction_time', 'REAL DEFAULT 12.0'),
        ('abductor_relation', 'TEXT DEFAULT "stranger"'),
        ('region_type', 'TEXT DEFAULT "Urban"'),
        ('population_density', 'INTEGER DEFAULT 5000'),
        ('missing_date', 'DATE')
    ]
    
    for col_name, col_type in columns:
        try:
            cursor.execute(f'ALTER TABLE missing_child ADD COLUMN {col_name} {col_type}')
            print(f"Added column {col_name}")
        except sqlite3.OperationalError as e:
            print(f"Column {col_name} might already exist: {e}")
            
    conn.commit()
    conn.close()

if __name__ == '__main__':
    update_db()
