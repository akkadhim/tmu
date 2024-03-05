import psycopg2
import array
from io import StringIO

class DB:
    def __init__(self, db_name, db_user, db_password, db_host, db_port):
        self._db_name = db_name
        self._conn = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        with open(db_name, 'r', encoding='ISO-8859-1') as f:
            buffer = StringIO(f.read())
        self._cursor = self._conn.cursor()
        self._cursor.execute(buffer.getvalue())
        self._cursor.close()

    def get_clauses_by_token(self, token, target_value, accumulation):
        if target_value == 1:
            operator = '> 0'
        else:
            operator = '< 0'

        self._cursor = self._conn.cursor()
        self._cursor.execute(
            '''SELECT weight, literals FROM clauses WHERE token = %s AND weight ''' + operator + ''' ORDER BY RANDOM() LIMIT %s''',
            (int(token), accumulation)
        )
        clauses = self._cursor.fetchall()
        self._cursor.close()
        return clauses
    
    def close_connection(self):
        self._conn.close()