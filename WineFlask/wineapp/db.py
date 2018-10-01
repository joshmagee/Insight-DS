#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database connection somehwat simplified.

We will initialize the database as if we are going to query it directly.
Instead, we will just connect it to a special object referenced for each request.
Later when we go to query the database, we will just return the dataframe directly.

@author: joshmagee
Thu Sep 27 08:31:39 2018
"""

import click
import pandas as pd
from flask import g
from sqlalchemy import create_engine
from flask.cli import with_appcontext

def init_db():
    db = get_db()
    return db
#    with current_app.open_resource('schema.sql') as f:
#        db.executescript(f.read().decode('utf8'))


@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

#when initializing app, make sure no previous connections exist
def init_app(app):
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)

def get_db():
    if 'db' not in g:
        g.db = create_engine('sqlite:///wine_data_clean.db')
    return g.db


def close_db(e=None):
    g.pop('db', None)
    #db = g.pop('db', None)
    #the following is only for sqlite
    #if db is not None:
    #    db.close()


def get_df():
    df = pd.read_sql_table('wine', con=g.db)
    return df


