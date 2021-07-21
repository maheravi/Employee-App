from sqlite3 import connect
from datetime import datetime


class Database:

    @staticmethod
    def insert(name, lastname, nationalcode, dateofbirth):
        my_con = connect('ListofEmployee.db')
        my_cursor = my_con.cursor()
        my_cursor.execute(f"INSERT INTO List_of_Employee(Name, LastName, NationalCode, DateofBirth, Pic) VALUES('{name}', '{lastname}', '{nationalcode}', '{dateofbirth}', '{str(nationalcode)+'.png'}')")
        my_con.commit()
        my_con.close()
        return True

    @staticmethod
    def select():
        my_con = connect('ListofEmployee.db')
        my_cursor = my_con.cursor()
        my_cursor.execute("SELECT * FROM List_of_Employee")
        result = my_cursor.fetchall()
        my_con.close()
        return result

    @staticmethod
    def delete(id):
        my_con = connect('ListofEmployee.db')
        my_cursor = my_con.cursor()
        my_cursor.execute(f"DELETE FROM List_of_Employee WHERE NationalCode = '{id}'")
        my_con.commit()
        my_con.close()

    @staticmethod
    def update():
        pass
