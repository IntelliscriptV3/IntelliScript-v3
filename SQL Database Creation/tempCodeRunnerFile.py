if __name__ == '__main__':
    Base.metadata.drop_all(engine)  # drops all tables
    Base.metadata.create_all(engine)
    seed_all()