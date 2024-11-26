from datetime import datetime

# Log events
def log_event(LOG_FILE, name):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.now()} - {name}\n")