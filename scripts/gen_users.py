import bcrypt

users = [
    ('rriftest1', 'Savjetnik01!', 'RRIF Test 1'),
    ('rriftest2', 'Savjetnik02!', 'RRIF Test 2'),
    ('rriftest3', 'Savjetnik03!', 'RRIF Test 3'),
]

for username, password, name in users:
    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    print(f"INSERT INTO users (username, password_hash, name) VALUES ('{username}', '{h}', '{name}');")
