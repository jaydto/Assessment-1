from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Make sure autoincrement=True is set
    uid = Column(String)  # Unique identifier
    password = Column(String)  # User's password
    first_name = Column(String)
    last_name = Column(String)
    username = Column(String)  # Username
    email = Column(String)
    avatar = Column(String)  # URL to the user's avatar image
    gender = Column(String)
    phone_number = Column(String)  # User's phone number
    social_insurance_number = Column(String)  # Social insurance number
    date_of_birth = Column(String)
    employment = Column(String)  # Employment details
    address = Column(String)  # User's address
    credit_card = Column(String)  # Credit card information
    subscription = Column(String)  # Subscription type (e.g., Premium, Basic)

