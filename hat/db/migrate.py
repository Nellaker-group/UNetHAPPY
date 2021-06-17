from playhouse.migrate import *


def main():
    db = SqliteDatabase("main.db")
    migrator = SqliteMigrator(db)

    slides_dir_field = TextField(default="")
    migrate(
        migrator.add_column('lab', 'slides_dir', slides_dir_field)
    )

    migrate(
        migrator.drop_column('slide', 'path')
    )





if __name__ == '__main__':
    main()