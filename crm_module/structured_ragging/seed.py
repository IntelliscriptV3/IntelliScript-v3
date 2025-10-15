"""
Seed script for your PostgreSQL DB using SQLAlchemy ORM.

Usage:
  1. Install dependencies: pip install sqlalchemy psycopg2-binary
  2. Set DATABASE_URL environment variable, e.g.
       export DATABASE_URL="postgresql+psycopg2://user:pass@localhost:5432/intelliscript"
  3. Run: python postgres_seed_sqlalchemy.py

What the script does:
  - Creates users for admins, teachers, students
  - Creates courses and schedules (ensures OL and AL classes don't clash inside their stream)
  - Enrolls 30 students per course (creates total unique students = 30 * 27)
  - Generates 6 months of attendance records up to 2025-09-26
  - Generates 6 months of monthly fees records
  - Generates weekly assessments for each week in that 6-month window and result rows for each enrolled student
  - Creates minimal ChatLogs and KnowledgeBase rows

Note: this script assumes your DB schema (tables and ENUM types) already exist and matches the names in your provided DDL.
"""

import os
import random
from datetime import datetime, date, time, timedelta
from collections import defaultdict

from sqlalchemy import (create_engine, Column, Integer, String, Text, Date, DateTime, Time,
                        Boolean, DECIMAL, ForeignKey, Enum, UniqueConstraint)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# -------------------- config --------------------
# DATABASE_URL = os.environ.get('DATABASE_URL')
DATABASE_URL="postgresql+psycopg2://postgres:5737@localhost:5433/intelliscript2"
if not DATABASE_URL:
    raise RuntimeError('Set DATABASE_URL environment variable, e.g. postgresql+psycopg2://user:pass@host:port/db')

engine = create_engine(DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# -------------------- ORM models (map to your existing tables) --------------------
# Note: we only include fields we will use for seeding
class User(Base):
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    email = Column(String(150), unique=True)
    role = Column(String)  # stored as enum in DB; keep string here
    password_hash = Column(String(255))
    created_at = Column(DateTime)

class Student(Base):
    __tablename__ = 'students'
    student_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    fname = Column(String(100))
    mname = Column(String(100))
    lname = Column(String(100))
    contact_no = Column(String(100))
    address_line1 = Column(String(100))
    address_line2 = Column(String(100))
    address_line3 = Column(String(100))
    age = Column(Integer)
    stream = Column(String)
    grade = Column(String)

class Teacher(Base):
    __tablename__ = 'teachers'
    teacher_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    fname = Column(String(100))
    mname = Column(String(100))
    lname = Column(String(100))
    contact_no = Column(String(100))
    address_line1 = Column(String(100))
    address_line2 = Column(String(100))
    address_line3 = Column(String(100))
    age = Column(Integer)

class Course(Base):
    __tablename__ = 'courses'
    course_id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, ForeignKey('teachers.teacher_id'))
    course_code = Column(String(20))
    subject = Column(String(100))
    grade = Column(String)
    stream = Column(String)
    day_of_week = Column(String)
    start_time = Column(Time)
    end_time = Column(Time)
    class_room = Column(String(100))

class Enrollment(Base):
    __tablename__ = 'enrollments'
    enrollment_id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey('students.student_id'))
    course_id = Column(Integer, ForeignKey('courses.course_id'))

class ChatLog(Base):
    __tablename__ = 'chat_logs'
    chat_id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    confidence_score = Column(DECIMAL(4,3))
    status = Column(String)
    created_at = Column(DateTime)

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_base'
    kb_id = Column(Integer, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    answered_by = Column(Integer)
    created_at = Column(DateTime)

class AdminQueue(Base):
    __tablename__ = 'admin_queue'
    queue_id = Column(Integer, primary_key=True)
    chat_id = Column(Integer)
    question = Column(Text)
    answer = Column(Text)
    answered_by = Column(Integer)
    assigned_at = Column(DateTime)

class Attendance(Base):
    __tablename__ = 'attendance'
    attendance_id = Column(Integer, primary_key=True)
    student_id = Column(Integer)
    course_id = Column(Integer)
    attendance_date = Column(Date)
    status = Column(String)
    remarks = Column(Text)
    created_at = Column(DateTime)

class Fee(Base):
    __tablename__ = 'fees'
    fee_id = Column(Integer, primary_key=True)
    student_id = Column(Integer)
    course_id = Column(Integer)
    month_year = Column(Date)
    amount = Column(DECIMAL(10,2))
    paid = Column(Boolean)
    paid_at = Column(DateTime)
    payment_method = Column(String(50))

class Assessment(Base):
    __tablename__ = 'assessments'
    assessment_id = Column(Integer, primary_key=True)
    course_id = Column(Integer)
    title = Column(String(150))
    description = Column(Text)
    max_marks = Column(Integer)
    assessment_date = Column(Date)

class AssessmentResult(Base):
    __tablename__ = 'assessment_results'
    result_id = Column(Integer, primary_key=True)
    assessment_id = Column(Integer)
    student_id = Column(Integer)
    marks_obtained = Column(Integer)
    graded_at = Column(DateTime)

# -------------------- helper functions --------------------
DOW = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)

# Map DOW string to python weekday int (Mon=0..Sun=6)
DOW_TO_INT = {name: i for i, name in enumerate(DOW)}
INT_TO_DOW = {i: name for name, i in DOW_TO_INT.items()}

# -------------------- seeding logic --------------------

first_teacher_names = [
    "Alice", "Benjamin", "Sophia", "Daniel", "Olivia", "James", "Isabella", "Liam",
    "Mia", "Ethan", "Charlotte", "Alexander", "Amelia", "Henry", "Grace", "Michael", "Emily"
]

last_teacher_names = [
    "Johnson", "Carter", "Miller", "Smith", "Brown", "Wilson", "Davis", "Anderson",
    "Thompson", "Martinez", "Garcia", "Taylor", "Moore", "Thomas", "Lee", "White", "Harris"
]

# Student names - 480 unique combinations
first_student_names = [
    "Aiden", "Ava", "Aaron", "Abigail", "Adam", "Addison", "Adrian", "Alexa", "Alan", "Alice",
    "Albert", "Allison", "Alex", "Amanda", "Alexander", "Amber", "Andre", "Amy", "Andrew", "Andrea",
    "Anthony", "Angela", "Antonio", "Anna", "Apollo", "Annie", "Arthur", "Ashley", "Austin", "Audrey",
    "Benjamin", "Bella", "Blake", "Brooke", "Brandon", "Brittany", "Brian", "Brianna", "Bruce", "Brooklyn",
    "Caleb", "Chloe", "Cameron", "Claire", "Carlos", "Camila", "Carter", "Caroline", "Charles", "Catherine",
    "Christian", "Charlotte", "Christopher", "Chloe", "Cole", "Christina", "Connor", "Clara", "Cooper", "Courtney",
    "Daniel", "Daisy", "David", "Danielle", "Diego", "Diana", "Dominic", "Destiny", "Dylan", "Delilah",
    "Ethan", "Emma", "Eli", "Emily", "Elijah", "Eva", "Elliott", "Elizabeth", "Eric", "Ella",
    "Felix", "Faith", "Finn", "Fiona", "Frank", "Freya", "Frederick", "Felicity", "Fernando", "Flora",
    "Gabriel", "Grace", "Gavin", "Gabriella", "George", "Genesis", "Grant", "Gianna", "Gregory", "Giselle",
    "Harrison", "Hannah", "Henry", "Haley", "Hunter", "Harper", "Hugo", "Hailey", "Hayden", "Helen",
    "Ian", "Isabella", "Isaac", "Iris", "Isaiah", "Ivy", "Ivan", "Isabelle", "Irving", "Ingrid",
    "Jacob", "Jasmine", "Jackson", "Julia", "James", "Jenna", "Jason", "Jessica", "Javier", "Jennifer",
    "Kai", "Kaia", "Kevin", "Katherine", "Kyle", "Kayla", "Kenneth", "Kimberly", "Keith", "Kylie",
    "Liam", "Lily", "Logan", "Lucy", "Lucas", "Luna", "Luke", "Lauren", "Leo", "Layla",
    "Mason", "Mia", "Matthew", "Madison", "Michael", "Maya", "Miles", "Megan", "Max", "Melanie",
    "Nathan", "Natalie", "Nicholas", "Nicole", "Noah", "Nora", "Nolan", "Naomi", "Nicolas", "Nina",
    "Oliver", "Olivia", "Owen", "Oksana", "Oscar", "Ophelia", "Otis", "Olga", "Orlando", "Octavia",
    "Parker", "Penelope", "Patrick", "Paige", "Paul", "Piper", "Peter", "Peyton", "Philip", "Priscilla",
    "Quinton", "Quinn", "Quentin", "Queenie", "Quade", "Quora", "Quincy", "Quinta", "Quest", "Quilla",
    "Ryan", "Rachel", "Robert", "Rebecca", "Roman", "Riley", "Richard", "Rose", "Ryder", "Ruby",
    "Samuel", "Sophia", "Sebastian", "Sarah", "Sean", "Samantha", "Simon", "Savannah", "Steven", "Stella",
    "Thomas", "Taylor", "Timothy", "Tessa", "Tyler", "Trinity", "Trevor", "Tiffany", "Tristan", "Tiana",
    "Victor", "Victoria", "Vincent", "Violet", "Vivian", "Valerie", "Vladimir", "Vanessa", "Vernon", "Vera",
    "William", "Willow", "Walter", "Whitney", "Wesley", "Wendy", "Warren", "Wanda", "Wayne", "Winona",
    "Xavier", "Ximena", "Xander", "Xara", "Xerxes", "Xyla", "Xavi", "Xiomara", "Xzavier", "Xenia",
    "Zachary", "Zoe", "Zane", "Zara", "Zeus", "Zelda", "Zion", "Zuri", "Zander", "Zinnia",
    "Abel", "Aria", "Axel", "Aurora", "Angelo", "Autumn", "Andre", "Avery", "Antonio", "Adelaide",
    "Blake", "Brielle", "Bryce", "Bianca", "Byron", "Beatrice", "Bennett", "Bailey", "Bradford", "Beverly",
    "Caleb", "Cecilia", "Chase", "Celeste", "Clark", "Cassandra", "Clay", "Candice", "Cruz", "Carmen",
    "Dean", "Daphne", "Derek", "Delilah", "Donovan", "Delaney", "Drake", "Destiny", "Damon", "Dahlia",
    "Edgar", "Elena", "Edwin", "Elise", "Ellis", "Eleanor", "Emmett", "Eliana", "Ezra", "Esther",
    "Fletcher", "Francesca", "Ford", "Frances", "Francisco", "Fatima", "Franklin", "Fernanda", "Flynn", "Faye",
    "Garrett", "Gemma", "Gideon", "Georgia", "Graham", "Gillian", "Griffin", "Gracie", "Gunner", "Gabrielle",
    "Holden", "Hope", "Hudson", "Harmony", "Hayes", "Hazel", "Heath", "Heather", "Hector", "Heidi",
    "Ira", "Imogen", "Iker", "Isla", "Ibrahim", "Iris", "Ignacio", "Irene", "Idris", "Ivy",
    "Jace", "Jade", "Jasper", "Jacqueline", "Jude", "Jordyn", "Joel", "Joy", "Jonah", "Josephine",
    "Kane", "Kinsley", "Knox", "Kendra", "Kaden", "Kennedy", "Kieran", "Kira", "King", "Katelyn",
    "Lance", "Leah", "Landon", "Lillian", "Lawrence", "Lila", "Lennox", "Lydia", "Lewis", "Lola",
    "Marcus", "Maya", "Martin", "Melody", "Mitchell", "Mila", "Morgan", "Molly", "Malik", "Megan",
    "Nolan", "Naomi", "Neil", "Nadia", "Nash", "Natasha", "Nate", "Nora", "Nico", "Nevaeh",
    "Owen", "Olivia", "Oscar", "Ophelia", "Otto", "Opal", "Orlando", "Oriana", "Orion", "Octavia"
]

last_student_names = [
    "Adams", "Allen", "Anderson", "Baker", "Brown", "Clark", "Davis", "Evans", "Garcia", "Hall",
    "Harris", "Jackson", "Johnson", "Jones", "King", "Lewis", "Lopez", "Martin", "Miller", "Moore",
    "Nelson", "Perez", "Roberts", "Robinson", "Rodriguez", "Smith", "Taylor", "Thomas", "Thompson", "Turner",
    "Walker", "White", "Williams", "Wilson", "Wright", "Young", "Campbell", "Carter", "Collins", "Cooper",
    "Edwards", "Flores", "Gonzalez", "Green", "Hernandez", "Hill", "Lee", "Mitchell", "Parker", "Phillips",
    "Reed", "Rivera", "Rogers", "Ross", "Sanchez", "Scott", "Stewart", "Torres", "Ward", "Watson",
    "Wood", "Bailey", "Bell", "Bennett", "Brooks", "Bryant", "Butler", "Cook", "Cox", "Diaz",
    "Foster", "Gray", "Hayes", "Henderson", "Howard", "Hughes", "Jenkins", "Kelly", "Long", "Murphy",
    "Nguyen", "Ortiz", "Palmer", "Patterson", "Perry", "Peterson", "Powell", "Price", "Ramirez", "Reyes",
    "Richardson", "Ramos", "Russell", "Simmons", "Sullivan", "Washington", "Wells", "West", "Wheeler", "Woods",
    "Barnes", "Fisher", "Freeman", "Gordon", "Hart", "Harrison", "Kim", "Lawrence", "Marshall", "Mcdonald",
    "Meyer", "Montgomery", "Morris", "Murray", "Owens", "Pierce", "Porter", "Reynolds", "Stone", "Walters",
    "Warren", "Webb", "Wells", "Arnold", "Bishop", "Black", "Bradley", "Burns", "Carpenter", "Crawford",
    "Daniel", "Dunn", "Ellis", "Franklin", "Gibson", "Graham", "Griffin", "Hamilton", "Harvey", "Hunt",
    "Jordan", "Kennedy", "Lane", "Mason", "Mills", "Olson", "Pearson", "Reid", "Ruiz", "Shaw",
    "Spencer", "Stevens", "Tucker", "Wagner", "Walsh", "Welch", "Banks", "Barrett", "Bass", "Burke",
    "Caldwell", "Campos", "Carlson", "Casey", "Chapman", "Chavez", "Chen", "Cole", "Cortez", "Craig",
    "Cross", "Cunningham", "Curtis", "Davidson", "Dean", "Delgado", "Dennis", "Dixon", "Duncan", "Elliott",
    "Estrada", "Ferguson", "Fernandez", "Fleming", "Ford", "Fowler", "Fox", "Francis", "Frazier", "Fuller",
    "Garner", "George", "Gilbert", "Gill", "Gomez", "Grant", "Graves", "Gross", "Guerrero", "Gutierrez",
    "Hansen", "Hawkins", "Haynes", "Hicks", "Hoffman", "Holland", "Holmes", "Holt", "Hopkins", "Howell",
    "Jacobs", "James", "Jimenez", "Johns", "Johnston", "Kelley", "Lowe", "Lucas", "Lynch", "Maldonado",
    "Mann", "Manning", "Martinez", "Matthews", "Maxwell", "May", "Mccoy", "Mcgee", "Mckinney", "Medina",
    "Mendoza", "Miles", "Miranda", "Morrison", "Moss", "Newton", "Nichols", "Nielsen", "Noble", "Norman",
    "Nunez", "Oliver", "Osborne", "Owen", "Padilla", "Page", "Parks", "Parsons", "Patel", "Patrick",
    "Payne", "Pennington", "Perkins", "Peters", "Petersen", "Pham", "Phelps", "Pope", "Preston", "Quinn",
    "Ramsey", "Ray", "Rees", "Reeves", "Rhodes", "Rice", "Rich", "Richards", "Riley", "Rios",
    "Robbins", "Robertson", "Roman", "Rose", "Rosario", "Roy", "Salazar", "Sanders", "Sandoval", "Santos",
    "Savage", "Schneider", "Schultz", "Schwartz", "Sharp", "Shelton", "Shepherd", "Sherman", "Silva", "Simpson",
    "Sims", "Singh", "Snyder", "Soto", "Sparks", "Stanton", "Stephens", "Stern", "Stevenson", "Sutton",
    "Swanson", "Tate", "Terry", "Tran", "Underwood", "Valdez", "Vargas", "Vasquez", "Vaughn", "Vega",
    "Villanueva", "Vincent", "Wade", "Wallace", "Walton", "Wang", "Warner", "Watkins", "Weaver", "Webster",
    "Welch", "Werner", "Wilcox", "Wilder", "Wiley", "Wilkins", "Williamson", "Willis", "Winters", "Wise",
    "Wolfe", "Woodard", "Woodward", "Yates", "York", "Zhang", "Zimmerman", "Abbott", "Abel", "Abrams",
    "Acevedo", "Acosta", "Adkins", "Aguilar", "Albert", "Alexander", "Ali", "Allison", "Alvarado", "Alvarez",
    "Anthony", "Archer", "Armstrong", "Ashton", "Austin", "Avery", "Ayala", "Ayers", "Barber", "Barker",
    "Barr", "Barry", "Bates", "Battle", "Bauer", "Bean", "Beard", "Becker", "Benson", "Berg",
    "Berger", "Bernard", "Berry", "Best", "Bird", "Blackwell", "Blair", "Blake", "Blanchard", "Blankenship",
    "Boone", "Booth", "Bowen", "Bowman", "Boyd", "Bradford", "Brady", "Branch", "Bray", "Brennan",
    "Brewer", "Briggs", "Bright", "Brock", "Brooke", "Brown", "Bruce", "Buck", "Buckley", "Bullock",
    "Burch", "Burgess", "Burnett", "Bush", "Byrd", "Cabrera", "Cain", "Camacho", "Cameron", "Cannon",
    "Cardenas", "Carey", "Carr", "Carson", "Cash", "Castillo", "Castro", "Chambers", "Chan", "Chandler"
]

# print("Total first names:", len(first_student_names))
# print("Total last names:", len(last_student_names))

def seed_all():
    session = Session()
    try:
        # config per your request
        ol_grades = [str(g) for g in range(6, 12)]  # 6..11
        al_grades = ['12','13','R']

        # OL: 2 courses each (science, maths) per grade
        ol_subjects = ['Mathematics','Science']
        # AL: 5 courses each (Combined Maths, Physics, Chemistry, Biology, ICT)
        al_subjects = ['Combined Maths','Physics','Chemistry','Biology','ICT']

        # classrooms
        classrooms = [f'Room-{i+1}' for i in range(5)]

        # Create 5 admins
        admins = []
        for i in range(5):
            u = User(username=f'Admin_{i+1}', email=f'admin{i+1}@inst.edu', role='admin', password_hash='axxx', created_at=datetime.now())
            session.add(u)
            session.flush()  # get user_id
            admins.append(u)
        session.commit()

        # Create teachers unique per course: total 12 + 15 = 27 teachers
        teachers = []
        total_courses = []  # will store (subject, grade, stream)

        # OL courses
        for grade in ol_grades:
            for subj in ol_subjects:
                total_courses.append((subj, grade, 'OL'))
        # AL courses
        for grade in al_grades:
            for subj in al_subjects:
                total_courses.append((subj, grade, 'AL'))

        # create teacher users and teacher rows
        teacher_objs = []
        for i, (subj, grade, stream) in enumerate(total_courses):
            if i > 16:
                break
            u = User(username=f'teacher{i+1}', email=f'teacher{i+1}@inst.edu', role='teacher', password_hash='txxx', created_at=datetime.now())
            session.add(u)
            session.flush()
            t = Teacher(user_id=u.user_id, fname=first_teacher_names[i], lname=last_teacher_names[i], contact_no=f'+940{70000000+i%1000}', address_line1=f'No {i+1}', address_line2=f'st{i+1} road', address_line3=f'City {i+1}', age=random.randint(25,60))
            session.add(t)
            session.flush()
            teacher_objs.append((t, u))
        
        session.commit()

        # Create course schedule times ensuring no same-stream clashes.
        # We'll create available weekly slots separately for OL and AL.
        # OL needs 2-hour blocks, AL needs 3-hour blocks.
        # Build non-overlapping slots: pick day_of_week and start_time so that within a stream no two courses conflict.

        def make_slots(num_slots, duration_hours):
            # produce num_slots (day_of_week, start_time, end_time) non-overlapping within a week
            slots = []
            # for simplicity, create slots per day rotating, and within day stagger by duration
            # available start hours: 08:00, 10:30, 13:00, 15:30
            candidate_starts = [time(8,0), time(10,30), time(13,0), time(15,30)]
            day_idx = 0
            slot_idx = 0
            while len(slots) < num_slots:
                day = DOW[day_idx % 5]  # Mon-Fri only
                start = candidate_starts[slot_idx % len(candidate_starts)]
                # compute end
                sh = start.hour + duration_hours
                end = time(sh, start.minute)
                # naive check: prevent duplicates
                if (day, start, end) not in slots:
                    slots.append((day, start, end))
                slot_idx += 1
                if slot_idx % len(candidate_starts) == 0:
                    day_idx += 1
            return slots

        ol_slots = make_slots(sum(1 for _,_,s in total_courses if s=='OL'), duration_hours=2)
        al_slots = make_slots(sum(1 for _,_,s in total_courses if s=='AL'), duration_hours=3)

        # Create Course rows and keep mapping course_id -> info
        course_objs = []
        student_objs = []
        enrollments = []
        ol_index = 0
        al_index = 0
        for idx, (subj, grade, stream) in enumerate(total_courses):
            
            if stream == 'OL':
                dow, st, et = ol_slots[ol_index]
                ol_index += 1
                teacher_obj, _ = teacher_objs[idx]
            else:
                dow, st, et = al_slots[al_index]
                al_index += 1
                if subj == 'Combined Maths':
                    teacher_obj, _ = teacher_objs[12]
                elif subj == 'Physics':
                    teacher_obj, _ = teacher_objs[13]
                elif subj == 'Chemistry':
                    teacher_obj, _ = teacher_objs[14]
                elif subj == 'Biology':
                    teacher_obj, _ = teacher_objs[15]
                elif subj == 'ICT':
                    teacher_obj, _ = teacher_objs[16]
                    
            course_code = f'{subj[:3].upper()}_{grade}_{idx+1}'
            course = Course(teacher_id=teacher_obj.teacher_id, course_code=course_code, subject=subj,
                            grade=grade, stream=stream, day_of_week=dow, start_time=st, end_time=et,
                            class_room=random.choice(classrooms))
            session.add(course)
            session.flush()
            course_objs.append(course)
            for s in range(30):
                uname = f'Student_{idx*30+s+1}'
                u = User(username=uname, email=f'student{idx*30+s+1}@inst.edu', role='student', password_hash='sxxx', created_at=datetime.now())
                session.add(u)
                session.flush()
                # print("index", idx*30+s)
                # Use different combinations: cycle through first names and last names separately
                first_name_idx = (idx*30+s) % len(first_student_names)
                last_name_idx = (idx*13+s*7) % len(last_student_names)  # Different pattern for last names
                s = Student(user_id=u.user_id, fname=first_student_names[first_name_idx], lname=last_student_names[last_name_idx], contact_no=f'+940{77000000+idx*30+s+1%1000}', address_line1=f'No {idx*30+s+1}', address_line2=f'rd{idx*30+s+1} Road', address_line3=f'City {idx*30+s+1}', age= int(grade) + 5 if grade != 'R' else 20, stream=stream, grade=grade)
                session.add(s)
                session.flush()
                student_objs.append(s)

                enroll = Enrollment(student_id=s.student_id, course_id=course.course_id)
                session.add(enroll)
                session.flush()
                enrollments.append(enroll)
        session.commit()

        # Create Students: unique students total = 30 * num_courses

        # num_courses_count = len(course_objs)
        # total_students = 30 * num_courses_count
        # student_objs = []
        # for i in range(total_students):
        #     uname = f'Student_{i+1}'
        #     u = User(name=uname, email=f'student{i+1}@inst.edu', role='student', password_hash='sxxx', created_at=datetime.now())
        #     session.add(u)
        #     session.flush()
        #     # assign random grade and stream for the student might not matter but we can set to one of OL/AL
        #     # To keep simple, leave stream/grade NULL except fill with 'R' for some
        #     s = Student(user_id=u.user_id, contact_no=f'+940{77000000+i%1000}', address='Student Address', age=random.randint(12,20), stream=None, grade=None)
        #     session.add(s)
        #     session.flush()
        #     student_objs.append(s)
        # session.commit()

        # Enroll 30 students per course: to match "each course has 30 students enrolled"
        # We'll take students in contiguous blocks of 30 for each course to be deterministic.

        # enrollments = []
        # st_index = 0
        # for course in course_objs:
        #     for j in range(30):
        #         student = student_objs[st_index + j]
        #         enroll = Enrollment(student_id=student.student_id, course_id=course.course_id)
        #         session.add(enroll)
        #         enrollments.append(enroll)
        #     st_index += 30
        # session.commit()

        # Attendance & Fees & Assessments
        # 6 months window until 2025-09-26 inclusive
        end_date = date(2025, 9, 26)
        start_date = (end_date - timedelta(days=183))  # approx 6 months (roughly 26 weeks)

        # build mapping course_id -> list of enrolled student_ids
        course_enrolled = defaultdict(list)
        for en in session.query(Enrollment).all():
            course_enrolled[en.course_id].append(en.student_id)

        # For each course, generate weekly attendance on the course's day_of_week between start_date and end_date
        attendance_rows = []
        for course in session.query(Course).all():
            dow_str = course.day_of_week
            weekday = DOW_TO_INT.get(dow_str, 0)
            # iterate through dates and pick those matching weekday
            for single_date in daterange(start_date, end_date):
                if single_date.weekday() == weekday:
                    for sid in course_enrolled[course.course_id]:
                        # mark present with high probability
                        status = random.choices(['present','absent'], weights=[85,10])[0]
                        att = Attendance(student_id=sid, course_id=course.course_id, attendance_date=single_date,
                                         status=status, remarks=None, created_at=datetime.now())
                        session.add(att)
            session.commit()

        # Fees: monthly records for months that intersect the 6-month window
        # We'll create one fee row per (student, course) per month within the 6-month window
        # Determine month starts (set day to 1 for uniqueness)
        months = set()
        cur = date(start_date.year, start_date.month, 1)
        while cur <= end_date:
            months.add(cur)
            # next month
            if cur.month == 12:
                cur = date(cur.year+1, 1, 1)
            else:
                cur = date(cur.year, cur.month+1, 1)
        months = sorted(list(months))

        for course_id, sids in course_enrolled.items():
            for sid in sids:
                for m in months:
                    amount = 100.00 + random.randint(0,50)  # example fee
                    paid = random.choices([True, False], weights=[90, 10])[0]
                    paid_at = datetime.combine(m + timedelta(days=5), datetime.min.time()) if paid else None
                    f = Fee(student_id=sid, course_id=course_id, month_year=m, amount=amount, paid=paid, paid_at=paid_at, payment_method='cash' if paid else None)
                    session.add(f)
            session.commit()

        # Assessments: one per week per course within the date range
        for course in session.query(Course).all():
            # weekly on the same day
            dow_str = course.day_of_week
            weekday = DOW_TO_INT.get(dow_str, 0)
            for single_date in daterange(start_date, end_date):
                if single_date.weekday() == weekday:
                    asmt = Assessment(course_id=course.course_id, title=f'Weekly Test {single_date}', description='Weekly assessment', max_marks=100, assessment_date=single_date)
                    
                    session.add(asmt)
                    session.flush()
                    # create results for every enrolled student
                    for sid in course_enrolled[course.course_id]:
                        marks = random.randint(0,100)
                        ar = AssessmentResult(assessment_id=asmt.assessment_id, student_id=sid, marks_obtained=marks, graded_at=datetime.now())
                        session.add(ar)
                    session.commit()

        # Minimal ChatLogs & KnowledgeBase entries

        # for i in range(10):
        #     cl = ChatLog(user_id=random.choice([u.user_id for u in admins]), question=f'Question {i+1}', answer=f'Answer {i+1}', confidence_score=0.900, status='answered', created_at=datetime.now())
        #     session.add(cl)
        # session.commit()

        # # Some KnowledgeBase rows
        # for i in range(10):
        #     kb = KnowledgeBase(question=f'FAQ {i+1}', answer=f'FAQ answer {i+1}', answered_by=random.choice([a.user_id for a in admins]), created_at=datetime.now())
        #     session.add(kb)
        # session.commit()

        print('Seeding complete')
    except Exception as e:
        session.rollback()
        print('Error during seeding:', e)
        raise
    finally:
        session.close()

Base.metadata.drop_all(engine) 
Base.metadata.create_all(engine)

if __name__ == '__main__':
    
    seed_all()
