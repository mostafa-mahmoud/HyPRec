#!/usr/bin/env python
"""
This module will provide the functionalities for parsing the data.
"""
import MySQLdb

class DataParser(object):
    """
    A class for parsing given data_file.
    """

    def process():
        """
        Start processing the data.
        """
        pass

    def get_ratings():
        """
        @returns A dictionary of user_id to a list of paper_id, of the papers
                 this user rated.
        """
        pass

    @staticmethod
    def get_connection():
        """
        @returns a database connection.
        """
        db = MySQLdb.connect(host="localhost", user="root", passwd ="")
        return db

    @staticmethod
    def set_up_database():
        """
        Creates the mysql tables in the database
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        cursor.execute("create database if not exists sahwaka")
        cursor.execute("use sahwaka")
        cursor.execute("create table if not exists users(id int(11) not null auto_increment, primary key(id))")
        cursor.execute("create table if not exists articles(id int(11) not null auto_increment, abstract text not null, primary key(id))")
        cursor.execute("create table if not exists articles_users(id int(11) not null auto_increment, user_id int(11) not null, article_id int(11) not null, primary key(id))")
        cursor.execute("create table words_users(id int(11) not null auto_increment, article_id int(11) not null, word varchar(55), primary key(id))")

        pass

DataParser.set_up_database()




