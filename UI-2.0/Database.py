import os
import sqlite3

# Function to create the main database and tables
def create_main_database(database_name):
    if not os.path.exists(database_name):
        os.makedirs(database_name)

    main_db_path = os.path.join(database_name, 'main_database.db')

    connection = sqlite3.connect(main_db_path)
    c = connection.cursor()

    # Create a table to store the list of databases
    c.execute("""CREATE TABLE IF NOT EXISTS database_list (
            database_name text,
            location text,
            creation_date text
        )""")

    connection.commit()
    connection.close()


def create_retraining_database(database_name):
    if not os.path.exists(database_name):
        os.makedirs(database_name)

    main_db_path = os.path.join(database_name, 'retrain_images.db')

    connection = sqlite3.connect(main_db_path)
    c = connection.cursor()

    # Create a table to store the list of databases
    c.execute("""CREATE TABLE IF NOT EXISTS database_list (
            image_name text,
            is_microplastic bool,
            bounding_box text
        )""")

    connection.commit()
    connection.close()


def retrain_data(database_name, image_name, is_microplastic, bounding_box):
    microplastics_db_path = os.path.join(database_name, 'retrain_images.db')
    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    is_microplastic_send = int(is_microplastic) 
 
    c.execute("INSERT INTO database_list (image_name, is_microplastic, bounding_box) VALUES (?, ?, ?)",
              (image_name, is_microplastic_send, bounding_box))
    
    connection.commit()
    connection.close()

def count_rows_in_retraining_database(database_name):
    main_db_path = os.path.join(database_name, 'retrain_images.db')

    connection = sqlite3.connect(main_db_path)
    c = connection.cursor()

    # Execute a query to count the number of rows in the database_list table
    c.execute("SELECT COUNT(*) FROM database_list")

    # Fetch the count
    count = c.fetchone()[0]

    connection.close()

    return count

# Function to add a new database entry to the main database
def add_database_entry(database_name, location, creation_date):

    connection = sqlite3.connect('main_database.db')
    if not os.path.exists(database_name):
        os.makedirs(database_name)

    main_db_path = os.path.join(database_name, 'microplastic.db')

    c = connection.cursor()
    c.execute("INSERT INTO database_list VALUES (?, ?, ?)",
              (database_name, location, creation_date))
    connection.commit()
    connection.close()

# Function to create a new microplastics database and table
def create_microplastics_database(database_name):
    microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS microplastics (
            image_loc text,
            particle_name text,
            length real,
            width real,
            color text,
            shape text,
            magnification integer,
            note text
        )""")
    connection.commit()
    connection.close()

# Function to insert data into the microplastics table
def insert_data(database_name, image_loc, particle_name, length, width, color, shape, magnification, note):
    microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    c.execute("INSERT INTO microplastics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (image_loc, particle_name, length, width, color, shape, magnification, note))
    connection.commit()
    connection.close()

# Function to retrieve data from the microplastics table
def get_data(database_name):

    if ".db" not in database_name:
        microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    else:
        microplastics_db_path = database_name
        
    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    c.execute("SELECT * FROM microplastics")
    data = c.fetchall()
    connection.close()
    return data

def get_image_data(database_name, particle_name):
    microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    
    # Check if database file exists
    if not os.path.exists(microplastics_db_path):
        return None
    
    try:
        connection = sqlite3.connect(microplastics_db_path)
        c = connection.cursor()
        
        # Check if table exists
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='microplastics'")
        if not c.fetchone():
            connection.close()
            return None
        
        # First try to find by exact particle_name match (without extension)
        particle_name_no_ext = particle_name.split('.')[0]
        c.execute("SELECT * FROM microplastics WHERE particle_name = ?", (particle_name_no_ext,))
        image_data = c.fetchall()
        
        # If not found, try to find by image_loc containing the filename
        if not image_data:
            # Build the full path to search for
            full_image_path = os.path.join(database_name, particle_name)
            c.execute("SELECT * FROM microplastics WHERE image_loc = ?", (full_image_path,))
            image_data = c.fetchall()
        
        connection.close()
        return image_data
    except Exception as e:
        print(f"Error reading database: {e}")
        if 'connection' in locals():
            connection.close()
        return None

def update_all_data(database_name, particle_name, length, width, color, shape, magnification, note):

    if ".db" not in database_name:
        microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    else:
        microplastics_db_path = database_name

    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    c.execute("UPDATE microplastics SET length=?, width=?, color=?, shape=?, magnification=?, note=? WHERE particle_name=?",
              (length, width, color, shape, magnification, note, particle_name))
    connection.commit()
    connection.close()

def update_table_data(database_name, particle_name, length, width, color, shape, row_id):

    if ".db" not in database_name:
        microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    else:
        microplastics_db_path = database_name

    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    c.execute("UPDATE microplastics SET particle_name=?, length=?, width=?, color=?, shape=? WHERE ROWID=?",
              (particle_name, length, width, color, shape, row_id))
    connection.commit()
    connection.close()

def update_record_by_index(database_name, row_index, **kwargs):
    if ".db" not in database_name:
        microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    else:
        microplastics_db_path = database_name

    try:
        connection = sqlite3.connect(microplastics_db_path)
        c = connection.cursor()
        
        # Get all records first to find the specific row
        c.execute("SELECT ROWID, image_loc, particle_name FROM microplastics ORDER BY ROWID")
        rows = c.fetchall()
        
        if row_index >= len(rows):
            print(f"Row index {row_index} out of range (total rows: {len(rows)})")
            connection.close()
            return False

        actual_rowid = rows[row_index][0]
        old_image_loc = rows[row_index][1]
        old_particle_name = rows[row_index][2]

        valid_fields = ['particle_name', 'length', 'width', 'color', 'shape', 'magnification', 'note']
        update_fields = {k: v for k, v in kwargs.items() if k in valid_fields}
        
        if not update_fields:
            print(f"No valid fields to update for row {row_index}")
            connection.close()
            return False

        # If particle_name is being changed, rename the actual image file
        if 'particle_name' in update_fields and old_image_loc:
            new_particle_name = update_fields['particle_name']
            if os.path.exists(old_image_loc):
                # Get the file extension from the old file
                file_ext = os.path.splitext(old_image_loc)[1]
                # Build new image path with new particle name
                image_dir = os.path.dirname(old_image_loc)
                new_image_loc = os.path.join(image_dir, new_particle_name + file_ext)
                
                # Rename the actual file
                try:
                    os.rename(old_image_loc, new_image_loc)
                    # Update image_loc in the database as well
                    update_fields['image_loc'] = new_image_loc
                    valid_fields.append('image_loc')
                    print(f"Renamed file from {old_image_loc} to {new_image_loc}")
                except Exception as e:
                    print(f"Error renaming file: {e}")

        set_clause = ", ".join([f"{field}=?" for field in update_fields.keys()])
        values = list(update_fields.values()) + [actual_rowid]
        
        query = f"UPDATE microplastics SET {set_clause} WHERE ROWID=?"
        print(f"Updating row {row_index} (ROWID {actual_rowid}): {update_fields}")
        c.execute(query, values)
        
        connection.commit()
        connection.close()
        return True
        
    except Exception as e:
        print(f"Error updating record at row {row_index}: {e}")
        if 'connection' in locals():
            connection.close()
        return False

def clear_table_data(database_name):
    
    if ".db" not in database_name:
        microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    else:
        microplastics_db_path = database_name

    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    c.execute("UPDATE microplastics SET 'particle_name' = '', 'length' = '', 'width' = '', 'color' = '', 'shape' = ''")
    connection.commit()
    connection.close()

# Function to delete data from the microplastics table
def delete_data(database_name, particle_name):
    microplastics_db_path = os.path.join(database_name, 'microplastic.db')
    connection = sqlite3.connect(microplastics_db_path)
    c = connection.cursor()
    c.execute("DELETE FROM microplastics WHERE particle_name=?", (particle_name,))
    connection.commit()
    connection.close()


def main():
    create_retraining_database("E:/THESIS/PolyVision/UI")
    count_rows_in_retraining_database("E:/THESIS/PolyVision/UI")

if __name__ == "__main__":
    main()
