#!/usr/bin/env python
"""
This module will provide the functionalities for parsing the data.
"""
import MySQLdb
import json
import os
import csv
from itertools import izip

class DataParser(object):
    """
    A class for parsing given data_file.
    """
    @staticmethod
    def process():
        """
        Start processing the data.
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        DataParser.set_up_database(db, cursor)
        cursor.execute("use sahwaka")
        DataParser.import_articles(db, cursor)
        DataParser.import_citations(db, cursor)
        DataParser.import_words(db, cursor)
        DataParser.import_users(db, cursor)
        DataParser.clean_up(db, cursor)
        pass

    @staticmethod
    def get_ratings():
        """
        @returns A dictionary of user_id to a list of paper_id, of the papers
                 this user rated.
        """
        db = DataParser.get_connection()
        cursor = db.cursor()
        cursor.execute("use sahwaka")
        cursor.execute("set group_concat_max_len=100000")
        cursor.execute("select user_id, concat('[', group_concat(article_id separator ', '),']') from articles_users group by user_id")
        ratings_hash = {}
        for (user_id, json) in cursor:
            ratings_hash[int(user_id)] = json
        DataParser.clean_up(db, cursor)
        return ratings_hash

    @staticmethod
    def import_articles(db, cursor):
        """
        reads raw-data.csv and fills the articles table
        """
        print "*** Inserting Articles ***"
        first_line = True
        with open(os.path.dirname(os.path.realpath(__file__)) + "/../data/raw-data.csv") as f:
            reader = csv.reader(f, quotechar='"')
            for line in reader:
                if first_line:
                    first_line = False
                    continue
                id = line[0]
                title = line[1]
                abstract = line[4]
                cursor.execute("insert into articles(id, title, abstract) values(%s, \"%s\", \"%s\")", (id, title, abstract.replace("\"", "\\\"")))

        pass


    @staticmethod
    def import_citations(db, cursor):
        """
        reads citations.dat and inserts rows in the citations table
        """
        print "*** Inserting Citations ***"
        id = 1
        with open(os.path.dirname(os.path.realpath(__file__)) + "/../data/citations.dat") as f:
            for line in f:
                splitted = line.replace("\n", "").split(" ")
                num_citations = splitted[0]
                for i in xrange(1, int(num_citations)):
                    cursor.execute("insert into sahwaka.citations(article_id, cited_article_id) values (%s,%s)", (id, splitted[i]))
                id += 1
        pass

    @staticmethod
    def import_words(db, cursor):
        """
        reads mult.dat and vocabulary.dat to insert bag of words representation in words_articles
        """
        print "*** Inserting Words ***"
        base_dir = os.path.dirname(os.path.realpath(__file__)) 
        with open(base_dir + "/../data/mult.dat") as bag, open(base_dir + "/../data/vocabulary.dat") as vocab:
            for entry, word in izip(bag, vocab):
                entry = entry.strip()
                word = word.strip()
                splitted = entry.split(" ")
                num_words = int(splitted[0])
                for i in xrange(1, num_words):
                    article_to_count = splitted[i].split(":")
                    article_id = article_to_count[0]
                    count = article_to_count[1]
                    cursor.execute("insert into words_articles(article_id, count, word) values (%s, %s, %s)", (article_id, count, word))
        pass

    @staticmethod
    def import_users(db, cursor):
        """
        reads users.dat to insert entries in users and articles_users table
        """
        print "*** Inserting Users ***"
        id = 1
        with open(os.path.dirname(os.path.realpath(__file__)) + "/../data/users.dat") as f:
            for line in f:
                splitted = line.replace("\n", "").split(" ")
                num_articles = int(splitted[0])
                cursor.execute("insert into users(id) values(%s)" % id)
                for i in xrange(1, num_articles):
                    cursor.execute("insert into articles_users(user_id, article_id) values(%s, %s)", (id, splitted[i]))
                id += 1
        pass

    @staticmethod
    def get_config():
        """
        @returns a json representation of the config file
        """
        with open(os.path.dirname(os.path.realpath(__file__)) + '/../config/config.json') as data_file:
            data = json.load(data_file)
        return data

    @staticmethod
    def get_connection():
        """
        @returns a database connection.
        """
        config = DataParser.get_config()
        db = MySQLdb.connect(host= config["database"]["host"], user= config["database"]["user"], passwd = config["database"]["password"])
        return db

    @staticmethod
    def set_up_database(db, cursor):
        """
        Creates the mysql tables in the database
        """
        config = DataParser.get_config()
        cursor.execute("create database if not exists %s" % (config["database"]["database_name"]))
        cursor.execute("use sahwaka")
        cursor.execute("create table if not exists users(id int(11) not null auto_increment, primary key(id))")
        cursor.execute("create table if not exists articles(id int(11) not null auto_increment, abstract text character set utf8mb4 COLLATE utf8mb4_general_ci not null, title varchar(255) not null, primary key(id))")
        cursor.execute("create table if not exists articles_users(id int(11) not null auto_increment, user_id int(11) not null, article_id int(11) not null, primary key(id))")
        cursor.execute("create table if not exists words_articles(id int(11) not null auto_increment, article_id int(11) not null, count int(8) not null, word varchar(55) not null, primary key(id))")
        cursor.execute("create table if not exists citations(id int(11) not null auto_increment, article_id int(11) not null, cited_article_id int(11) not null, primary key(id))")
        cursor.execute("create table if not exists words(id int(11) not null, word varchar(55), primary key(id))")
        pass
    
    @staticmethod
    def clean_up(db, cursor):
        """
        Method is always called to make sure database connections are closed and committed
        """
        db.commit()
        cursor.close()
        db.close()


if __name__ == "__main__": DataParser.process()

