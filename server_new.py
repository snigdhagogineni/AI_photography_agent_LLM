from flask import Flask, request, jsonify, Response
import os, uuid, json, bcrypt, io
import requests, time, pytz
import threading
import base64, zipfile
from datetime import datetime
import mysql.connector
from mysql.connector import Error
import sendgrid
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
from PIL import Image
from flask_cors import CORS
import replicate
from threading import Lock
import shutil
import tempfile
from huggingface_hub import HfApi
from dotenv import load_dotenv

#environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)

#values from env variables
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
HF_TOKEN = os.getenv('HF_TOKEN')
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY')
DB_NAME = os.getenv('DB_NAME', 'database-1')  
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_USER = os.getenv('DB_USER')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT', '3306'))  
SENDGRID_FROM_EMAIL = "snigdhagogineni14@gmail.com"

#db
DB_CONFIG = {
    'host': DB_HOST,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'database': DB_NAME,
    'port': DB_PORT
}

#SSE 
#sse_clients = {}
#sse_lock = Lock()

def get_db_connection():
    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT
    )
    connection.autocommit = True
    return connection

def init_db():
    conn = get_db_connection()
    if not conn:
        print("Failed to initialize database")
        return 
    cursor = conn.cursor()
    
    #users table 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        user_id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL UNIQUE,
        email VARCHAR(255) NOT NULL UNIQUE,
        password VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB
    ''')
    
    #images table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        image_id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        image_name VARCHAR(255) NOT NULL,
        image_data LONGBLOB NOT NULL,
        description TEXT,
        category VARCHAR(255) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    ) ENGINE=InnoDB
    ''')
    
    #projects table 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        project_id VARCHAR(36) PRIMARY KEY,
        user_id INT NOT NULL,
        project_name VARCHAR(255) NOT NULL,
        status VARCHAR(50) DEFAULT 'uploaded',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        model_id VARCHAR(255),
        metadata TEXT,
        model_type VARCHAR(50) DEFAULT 'person_couple',
        product_info TEXT,
        training_data LONGBLOB,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
        CONSTRAINT chk_model_type CHECK (model_type IN ('person_couple', 'product_celebrity'))
    ) ENGINE=InnoDB
    ''')
    
    #index if it doesn't exist
    try:
        cursor.execute('CREATE INDEX idx_users_username ON users(username)')
    except mysql.connector.Error as err:
        if err.errno == 1061: #error code for duplicate key
            pass #index present, ignore error
        else:
            print(f"Error creating index: {err}")
    conn.commit()
    cursor.close()
    conn.close()

def add_training_data_column():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists 
        cursor.execute(f"""
        SELECT COUNT(*)
        FROM information_schema.COLUMNS 
        WHERE TABLE_SCHEMA = '{DB_NAME}'
        AND TABLE_NAME = 'projects'
        AND COLUMN_NAME = 'training_data'
        """)
        
        if cursor.fetchone()[0] == 0:
            cursor.execute("ALTER TABLE projects ADD COLUMN training_data LONGBLOB")
            conn.commit() 
        else:
            print("training_data col already exists") 
    except Exception as e:
        print(f"Error adding training_data column: {str(e)}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

#initialize db
init_db()
add_training_data_column()

@app.route("/", methods=["GET"])
def health_check():
    return {"status": "ok"}, 200

#register
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    if not all([data.get("username"), data.get("email"), data.get("password")]):
        return {"error": "Missing required fields"}, 400
    
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}, 500
        
        cursor = conn.cursor(dictionary=True)
        
        #check if user exists
        cursor.execute("SELECT username FROM users WHERE email = %s OR username = %s", 
                      (data["email"], data["username"]))
        if cursor.fetchone():
            return {"error": "Email or username already registered"}, 400
        
        password = data["password"].encode('utf-8')
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
        
        cursor.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (data["username"], data["email"], hashed_password)
        )
        conn.commit()
        
        return {"message": "Registration successful"}, 201
        
    except mysql.connector.Error as e:
        if e.errno == 1062: # Duplicate entry error
            return {"error": "Email or username already registered"}, 400
        return {"error": f"Registration failed: {str(e)}"}, 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

#login
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if not all([data.get("email"), data.get("password")]):
        return {"error": "Missing required fields"}, 400
    
    try:
        conn = get_db_connection()
        if not conn:
            return {"error": "Database connection failed"}, 500
        
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, username, password FROM users WHERE email = %s", (data["email"],))
        user = cursor.fetchone()
        
        if not user:
            return {"error": "Invalid credentials"}, 401
        
        column_names = [desc[0] for desc in cursor.description]
        user_dict = dict(zip(column_names, user))
        
        stored_password = user_dict["password"]
        if isinstance(stored_password, str):
            stored_password = stored_password.encode('utf-8')
        
        input_password = data["password"].encode('utf-8')
        
        if not bcrypt.checkpw(input_password, stored_password):
            return {"error": "Invalid credentials"}, 401
        
        return jsonify({
            "message": "Login successful",
            "username": user_dict["username"],
            "email": data["email"]
        }), 200
        
    except mysql.connector.Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Login failed: {str(e)}"}), 500
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

#upload images
@app.route("/upload_images", methods=["POST"])
def upload_images():
    username = request.form.get("username")
    if not username:
        return {"error": "Username is required"}, 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        #user_id
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        user_result = cursor.fetchone()
        if not user_result:
            return {"error": "User not found"}, 404
        
        user_id = user_result["user_id"]
        project_id = str(uuid.uuid4())
        project_name = f"Project_{project_id[:8]}"
        
        #files and metadata
        files = request.files.getlist("files")
        if len(files) != 15:
            return {"error": "Exactly 15 images required"}, 400
        
        #metadata structure
        metadata = {
            'files': [],
            'person1_code': request.form.get('person1_code', 'wxy'),
            'person2_code': request.form.get('person2_code', 'mno')}
        
        #store images in MySQL
        for i, file in enumerate(files):
            if not file.filename:
                continue
            
            desc = request.form.get(f"desc_{i}")
            category = request.form.get(f"category_{i}")
            if not desc or not category:
                return {"error": f"Missing metadata for image {i}"}, 400
            
            #read image data
            image_data = file.read()
            
            #insert image in db
            cursor.execute("""
            INSERT INTO images (user_id, image_name, image_data, description, category)
            VALUES (%s, %s, %s, %s, %s)
            """, (
                user_id,
                file.filename,
                image_data,
                desc,
                category
            ))
            
            #get image_id of inserted image
            image_id = cursor.lastrowid
            
            #add to metadata
            metadata['files'].append({ 'image_id': image_id,'description': desc,'category': category})
        
        #insert project
        cursor.execute("""
        INSERT INTO projects (project_id, user_id, project_name, status, metadata, model_type, product_info)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            project_id, 
            user_id, 
            project_name, 
            "uploaded",
            json.dumps(metadata),
            request.form.get('model_type', 'person_couple'),
            request.form.get('product_info', '{}')
        ))
        conn.commit()
        
        #start training in bg
        threading.Thread(target=start_training, args=(project_id,)).start()
        return {"message": "Upload successful", "project_id": project_id}, 200
        
    except Exception as e:
        print(f"Error in upload_images: {str(e)}")
        return {"error": f"Upload failed: {str(e)}"}, 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
#huggingface upload
def upload_to_huggingface(weights_url, hf_token, hf_repo):
    try:
        print("\n=== Starting HuggingFace Upload ===")
        print(f"Repository: {hf_repo}")
        print(f"Weights URL: {weights_url}")
        
        #temp dir to store downloaded file
        temp_dir = tempfile.mkdtemp()
        local_file_path = os.path.join(temp_dir, "model.safetensors")
        
        try:
            #download file
            print(" Downloading weights file...")
            response = requests.get(weights_url, stream=True)
            response.raise_for_status()
            
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(" File downloaded successfully") 
            #to HuggingFace
            print(" Uploading to HuggingFace...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_{timestamp}.safetensors"
            
            #use Hugging Face Hub lib if there, else fallback to direct API
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_file(
                    path_or_fileobj=local_file_path,
                    path_in_repo=f"models/{filename}",
                    repo_id=hf_repo,
                    token=hf_token
                )
                print(" Upload successful using HfApi")
                return f"https://huggingface.co/{hf_repo}/resolve/main/models/{filename}"
            except ImportError:
                #fallback to direct API
                headers = {"Authorization": f"Bearer {hf_token}"}
                upload_url = f"https://huggingface.co/api/repos/{hf_repo}/upload"
                
                with open(local_file_path, 'rb') as f:
                    files = {
                        'file': (f"models/{filename}", f, 'application/octet-stream')
                    }
                    response = requests.post(upload_url, headers=headers, files=files)
                
                if response.status_code in [200, 201]:
                    print(f" Upload successful: {response.status_code}")
                    return f"https://huggingface.co/{hf_repo}/resolve/main/models/{filename}"
                
                #original URL as fallback, doesnt fail
                return weights_url 
        except Exception as e:
            return weights_url #original URL as fallback
        finally:
            #clean temp dir 
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f" Error uploading to HuggingFace: {str(e)}")
        return weights_url #original URL as fallback
#sendgrid
def send_training_complete_email(to_email, project_name, hf_url=None):
    try:
        print(f"\n=== Sending completion email to {to_email} ===")
        
        content = f'''
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <p>Hey there!</p>
            <p>Your project "{project_name}" has finished training and is ready to use.</p>
            <p>You can now go to app and start generating images with your custom model.</p>
        '''
        if hf_url:
            content += f'''
            <p>Your model has been saved and is available at:</p>
            <p>Model URL: <a href="{hf_url}">{hf_url}</a></p>
            '''
        content += '''
            <p>Thank you for using our platform!</p>
            <p>Best regards,<br>The AI Image Platform</p>
        </div>
        '''
        
        message = Mail(
            from_email=SENDGRID_FROM_EMAIL,
            to_emails=to_email,
            subject='Training Complete - AI Image Platform',
            html_content=content
        )
        
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f" Email sent successfully: {response.status_code}")
        return True
        
    except Exception as e:
        print(f" Failed to send email: {str(e)}")
        return False

#training
def start_training(project_id):
    print(f"\n=== Starting Training for Project: {project_id} ===")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        print("\n Initialized Replicate client")
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            #project details
            cursor.execute("""
            SELECT p.project_name, u.username, p.metadata, u.email, p.model_type, p.product_info, u.user_id
            FROM projects p
            JOIN users u ON p.user_id = u.user_id
            WHERE p.project_id = %s
            """, (project_id,))
            project = cursor.fetchone()
            
            if not project:
                raise ValueError("Project not found")
            
            #update status to training
            cursor.execute("UPDATE projects SET status = %s WHERE project_id = %s", ("training", project_id))
            conn.commit()
            
            metadata_dict = json.loads(project['metadata']) if isinstance(project['metadata'], str) else (project['metadata'] or {})
            
            print(f"\n✓ Project details retrieved:")
            print(f" - Name: {project['project_name']}")
            print(f" - Type: {project['model_type']}")
            print(f" - User: {project['username']}")
            
            #create temp dir for zip file
            temp_dir = f"temp_{project_id}_{timestamp}"
            os.makedirs(temp_dir, exist_ok=True)
            
            #process images from metadata
            files_processed = 0
            for file_data in metadata_dict.get("files", []):
                #img data from MySQL
                cursor.execute("""
                SELECT image_data, category, description 
                FROM images 
                WHERE image_id = %s AND user_id = %s
                """, (file_data.get('image_id'), project['user_id']))
                image_result = cursor.fetchone()
                
                if not image_result:
                    continue
                
                try:
                    #save image to temp dir
                    filename = f"{image_result['category']}_{files_processed}.jpg"
                    file_path = os.path.join(temp_dir, filename)
                    
                    with open(file_path, 'wb') as f:
                        f.write(image_result['image_data'])
                    
                    #caption file
                    if image_result['description']:
                        caption_name = os.path.splitext(filename)[0] + ".txt"
                        caption_path = os.path.join(temp_dir, caption_name)
                        with open(caption_path, 'w') as f:
                            f.write(image_result['description'])
                    
                    files_processed += 1
                    print(f" Processed image {files_processed}/15")
                
                except Exception as e:
                    print(f"Error processing image: {str(e)}")
                    continue
            
            if files_processed != 15:
                raise ValueError(f"Expected 15 images but processed {files_processed}")
            
            #zip file
            zip_path = os.path.join(temp_dir, "training.zip")
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for file in os.listdir(temp_dir):
                    if file != "training.zip":
                        zipf.write(os.path.join(temp_dir, file), file)
            
            print(f" Created training zip with {files_processed} images and captions")
            
            #training params as provided
            training_params = {
                "steps": 1000,
                "network_dim": 32,
                "network_alpha": 32,
                "lora_rank": 64,
                "train_unet": True,
                "train_text_encoder": True,
                "gradient_checkpointing": True,
                "mixed_precision": "fp16",
                "scheduler": "ddpm",
                "optimizer": "adamw8bit",
                "learning_rate": "0.0001",
                "resolution": "512,768,1024",
                "batch_size": 1,
                "gradient_accumulation_steps": 4,
                "clip_skip": 2,
                "save_every_n_steps": 100,
                "save_model_as": "safetensors",
                "shuffle_tags": False,
                "keep_tokens": 4,
                "caption_extension": ".txt",
                "wandb_project": "flux_train_replicate",
                "wandb_save_interval": 100,
                "wandb_sample_interval": 100,
                "caption_dropout_rate": "0.0",
                "cache_latents_to_disk": True,
                "autocaption": False,
                "caption_prefix": "RAW photo, professional portrait photograph, photorealistic, realistic, real person, detailed face, sharp focus on face, high detail face, perfect facial features, detailed facial features, perfect eyes, detailed eyes, perfect skin, cinematic lighting, 8k uhd, masterpiece, best quality, intricate details, "
            }
            
            #trigger word based on model type
            if project['model_type'] == "person_couple":
                person1_code = metadata_dict.get('person1_code', 'wxy')
                person2_code = metadata_dict.get('person2_code', 'mno')
                trigger_word = f"{person1_code}_{person2_code}"
            else:
                product_info_dict = json.loads(project['product_info']) if isinstance(project['product_info'], str) else (project['product_info'] or {})
                product_name = product_info_dict.get('product_name', 'product')
                trigger_word = f"celebrity_{product_name}"
            
            training_params["trigger_word"] = trigger_word
            metadata_dict['trigger_word'] = trigger_word
            
            print("\n Starting training on Replicate...")
            print(f" - Trigger word: {trigger_word}")
            
            #submit training to Replicate
            try:
                with open(zip_path, 'rb') as f:
                    #create training with inp_img
                    training = client.trainings.create(
                        version="ostris/flux-dev-lora-trainer:b6af14222e6bd9be257cbc1ea4afda3cd0503e1133083b9d1de0364d8568e6ef",
                        input={
                            **training_params,
                            "input_images": open(zip_path, 'rb')
                        },
                        destination="snigdhagogineni/photo_ai")
                
                training_id = training.id
                print(f"\n Training started successfully")
                print(f" - Training ID: {training_id}")
                
                #store full model ID in for generate_img
                full_model_id = f"snigdhagogineni/photo_ai:{training_id}"
                print(f" - Full Model ID: {full_model_id}")
                
                #store zip file in db
                with open(zip_path, 'rb') as zip_file:
                    zip_data = zip_file.read()
                    cursor.execute("""
                    UPDATE projects 
                    SET model_id = %s,
                    metadata = %s,
                    training_data = %s
                    WHERE project_id = %s
                    """, (
                        full_model_id,
                        json.dumps(metadata_dict),
                        zip_data,
                        project_id
                    ))
                conn.commit()
            except Exception as e:
                print(f"Error submitting training: {str(e)}")
                cursor.execute("""
                UPDATE projects 
                SET status = %s, 
                metadata = %s
                WHERE project_id = %s
                """, (
                    "failed",
                    json.dumps({"error": str(e), **metadata_dict}),
                    project_id
                ))
                conn.commit()
                raise ValueError(f"Failed to submit training: {str(e)}")
            
            #monitor training progress
            while True:
                training = client.trainings.get(training_id)
                status = training.status
                
                if status == "succeeded":
                    output = training.output
                    if not output:
                        print("Warning: Training output is empty, but continuing...")
                        output = {}
                    
                    # Get weights URL
                    weights_url = None
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if isinstance(value, str) and ('replicate.delivery' in value or '.safetensors' in value):
                                weights_url = value
                                break
                    
                    if not weights_url:
                        print("Warning: No weights URL found in training output, but continuing...")
                        # Use a fallback URL or the training ID
                        weights_url = f"https://replicate.com/p/{training_id}"
                    
                    print(f"\n Training completed successfully")
                    print(f" - Weights URL: {weights_url}")
                    
                    # Try to upload to HuggingFace, but continue even if it fails
                    hf_url = None
                    try:
                        print("\n Uploading model to HuggingFace...")
                        hf_url = upload_to_huggingface(
                            weights_url=weights_url,
                            hf_token=HF_TOKEN,
                            hf_repo="snigdhagogineni/photo_ai"
                        )
                        print(f" - HuggingFace URL: {hf_url}")
                    except Exception as e:
                        print(f"\n⚠️ HuggingFace upload failed: {str(e)}")
                        hf_url = weights_url #replicate URL as fallback
                    
                    #use a valid URL - either HF/weights URL
                    model_url = hf_url or weights_url
                    
                    #metadata and project status
                    metadata_dict.update({
                        'huggingface_url': model_url,
                        'weights_url': weights_url,
                        'version_id': training_id,
                        'full_model_path': full_model_id,
                        'training_output': output,
                        'status': 'completed' #add status
                    })
                    
                    #update db to put project as completed
                    cursor.execute("""
                    UPDATE projects 
                    SET status = %s, 
                    model_id = %s, 
                    metadata = %s
                    WHERE project_id = %s
                    """, (
                        "completed",
                        full_model_id,
                        json.dumps(metadata_dict),
                        project_id
                    ))
                    conn.commit()
                    
                    #send email with either HuggingFace URL or Replicate URL
                    print(f"\n✓ Sending completion email to {project['email']}...")
                    email_sent = send_training_complete_email(project['email'], project['project_name'], model_url)
                    if email_sent:
                        print(" - Email sent successfully")
                    else:
                        print(" - Failed to send email, but continuing...")
                    
                    print("\n✓ Training process completed successfully")
                    break
                
                elif status == "failed":
                    error_message = getattr(training, 'error', 'Unknown error')
                    print(f"\n Training failed: {error_message}")
                    
                    #project status to failed
                    cursor.execute("""
                    UPDATE projects 
                    SET status = %s, 
                    metadata = %s
                    WHERE project_id = %s
                    """, (
                        "failed",
                        json.dumps({**metadata_dict, 'error': error_message}),
                        project_id
                    ))
                    conn.commit()
                    
                    raise ValueError(f"Training failed: {error_message}")
                
                #canceled or other status project
                elif status not in ["starting", "processing"]:
                    print(f"\n⚠️ Unknown status: {status}")
                    
                    if status == "canceled":
                        cursor.execute("""
                        UPDATE projects 
                        SET status = %s, 
                        metadata = %s
                        WHERE project_id = %s
                        """, (
                            "failed",
                            json.dumps({**metadata_dict, 'error': f"Training {status}"}),
                            project_id
                        ))
                        conn.commit()
                    
                    raise ValueError(f"Training failed: {status}")
                time.sleep(30)
        
        except Exception as e:
            print(f"\n Training error: {str(e)}")
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("""
                UPDATE projects 
                SET status = %s, 
                metadata = %s 
                WHERE project_id = %s
                """, (
                    "failed",
                    json.dumps({"error": str(e)}),
                    project_id
                ))
                conn.commit()
            except Exception as db_error:
                print(f"\n Failed to update project status: {str(db_error)}")
        finally:
            cursor.close()
            conn.close()
        
        print("\n=== Training Process Complete ===")

    except Exception as e:
        print(f"\n Training error: {str(e)}")
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
            UPDATE projects 
            SET status = %s, 
            metadata = %s 
            WHERE project_id = %s
            """, (
                "failed",
                json.dumps({"error": str(e)}),
                project_id
            ))
            conn.commit()
        except Exception as db_error:
            print(f"\n Failed to update project status: {str(db_error)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
        
        print("\n=== Training Process Complete ===")

def convert_to_cst(utc_dt):
    #convert UTC to CST
    if not utc_dt:
        return None
    utc = pytz.timezone('UTC')
    cst = pytz.timezone('America/Chicago')
    return utc.localize(utc_dt).astimezone(cst)

#trained models
@app.route("/get_trained_models", methods=["GET"])
def get_trained_models():
    username = request.args.get("username")
    model_type = request.args.get("model_type", "all")
    
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    print(f"\n=== Fetching trained models for user: {username}, model_type: {model_type} ===")
    
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
        
        cursor = conn.cursor(dictionary=True)
        
        #user_id first
        cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
        user_result = cursor.fetchone()
        if not user_result:
            print(f"User not found: {username}")
            return jsonify({"error": "User not found"}), 404
        
        user_id = user_result['user_id']
        print(f"Found user_id: {user_id}")
        
        #get models -completed
        if model_type == "all":
            query = """
                SELECT project_id, project_name, model_id, model_type,
                metadata, product_info, created_at, status
                FROM projects 
                WHERE user_id = %s AND model_id IS NOT NULL AND status = 'completed'
                ORDER BY created_at DESC
            """
            cursor.execute(query, (user_id,))
        else:
            query = """
                SELECT project_id, project_name, model_id, model_type,
                metadata, product_info, created_at, status
                FROM projects 
                WHERE user_id = %s AND model_id IS NOT NULL AND status = 'completed' AND model_type = %s
                ORDER BY created_at DESC
            """
            cursor.execute(query, (user_id, model_type))
        
        rows = cursor.fetchall()
        print(f"Found {len(rows)} completed models")
        
        projects = []
        for row in rows:
            try:
                #skipping projects without model_id
                if not row['model_id']:
                    print(f"Skipping project {row['project_id']} - no model_id")
                    continue
                
                #check model_id in correct format
                model_id = row['model_id']
                if ":" not in model_id:
                    print(f"Skipping project {row['project_id']} - invalid model_id format: {model_id}")
                    continue
                
                #created_at to CST
                created_at_utc = row['created_at']
                created_at_cst = convert_to_cst(created_at_utc)
                
                #metadata is a string-parsing
                metadata = row['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                elif metadata is None:
                    metadata = {}
                
                #project data
                project_data = {
                    "project_id": row['project_id'],
                    "name": row['project_name'],
                    "model_id": model_id,
                    "model_type": row['model_type'],
                    "status": "completed",
                    "created_at": created_at_cst.strftime('%Y-%m-%d %H:%M:%S') if created_at_cst else None,
                    "metadata": metadata
                }
                
                #trigger word
                trigger_word = metadata.get('trigger_word')
                if not trigger_word:
                    if row['model_type'] == "person_couple":
                        person1_code = metadata.get('person1_code', 'wxy')
                        person2_code = metadata.get('person2_code', 'mno')
                        trigger_word = f"{person1_code}_{person2_code}"
                    else:
                        #product_info is string
                        product_info = row['product_info']
                        if isinstance(product_info, str):
                            try:
                                product_info = json.loads(product_info)
                            except json.JSONDecodeError:
                                product_info = {}
                        elif product_info is None:
                            product_info = {}
                        
                        product_name = product_info.get('product_name', 'product')
                        trigger_word = f"celebrity_{product_name}"
                
                project_data["trigger_word"] = trigger_word
                
                #product info for product_celebrity models
                if row['model_type'] == "product_celebrity" and row['product_info']:
                    product_info = row['product_info']
                    if isinstance(product_info, str):
                        try:
                            product_info = json.loads(product_info)
                        except json.JSONDecodeError:
                            product_info = {}
                    project_data["product_info"] = product_info
                
                print(f"Adding project: {project_data['name']} with model_id: {project_data['model_id']}")
                projects.append(project_data)
            
            except Exception as e:
                print(f"Error processing project {row['project_id']}: {str(e)}")
                continue
        
        print(f"Returning {len(projects)} models")
        return jsonify(projects)
    
    except Exception as e:
        print(f"Error in get_trained_models: {str(e)}")
        return jsonify({"error": f"Failed to fetch models: {str(e)}"}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

@app.route("/generate_image", methods=["POST"])
def generate_image():
    data = request.json
    model_id = data.get("model_id")
    prompt = data.get("prompt")
    username = data.get("username")
    
    print(f"\n=== Generating image with model: {model_id} ===")
    print(f" - Username: {username}")
    print(f" - Prompt: {prompt}")
    
    if not all([model_id, prompt, username]):
        return jsonify({"error": "Missing required fields"}), 400   
    try:   
        client = replicate.Client(api_token=REPLICATE_API_TOKEN)
        print(f" Initialized Replicate client")
        
        #model ID format 
        model_path, version_id = model_id.split(":")
        if model_path != "snigdhagogineni/photo_ai":
            print(f" Invalid model path: {model_path}")
            return jsonify({"error": "Invalid model path"}), 400
        
        print(f" Validated model ID format")
        print(f" - Model path: {model_path}")
        print(f" - Version ID: {version_id}")
        
        #model exists in db and get metadata
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        #get project details
        cursor.execute("""
            SELECT p.model_id, p.status, p.metadata, p.model_type, u.username
            FROM projects p
            JOIN users u ON p.user_id = u.user_id
            WHERE p.model_id = %s AND p.status = 'completed' AND u.username = %s
        """, (model_id, username))
        project = cursor.fetchone()
        
        if not project:
            print(f" Model not found in database or not completed: {model_id}")
            return jsonify({"error": "Model not found or training not completed"}), 404
        
        print(f" Found model in database")
        print(f" - Status: {project['status']}")
        print(f" - Model Type: {project['model_type']}")
        
        #parse metadata
        try:
            metadata = json.loads(project['metadata']) if isinstance(project['metadata'], str) else project['metadata']
            print(f" - Metadata: {json.dumps(metadata, indent=2)}")
            
            #trigger word from metadata
            trigger_word = metadata.get('trigger_word')
            if trigger_word and trigger_word not in prompt:
                print(f" - Adding trigger word '{trigger_word}' to prompt")
                prompt = f"{trigger_word}, {prompt}"
                print(f" - Final prompt: {prompt}")
        except Exception as e:
            print(f" Error parsing metadata: {str(e)}")
        
        #prediction with trained model
        print(f"\n - Running prediction with model: {model_id}")
        print(f" - Using prompt: {prompt}")
        
        #trained model
        try:
            metadata = json.loads(project['metadata']) if isinstance(project['metadata'], str) else project['metadata']
            if not metadata or 'training_output' not in metadata:
                return jsonify({"error": "Invalid model metadata"}), 400
            
            #correct version from training o/p
            training_output = metadata.get('training_output', {})
            model_version = training_output.get('version')
            
            if not model_version:
                return jsonify({"error": "Model version not found"}), 400
            
            print(f" - Using model version from training output: {model_version}")
            
            output = client.run(
                model_version,  #version from training o/p
                input={
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted, deformed, ugly, duplicate, double, multiple, wrong, bad anatomy",
                    "num_inference_steps": 40,
                    "guidance_scale": 7.5,
                    "width": 512,
                    "height": 768
                }
            )
            print(f" Prediction completed")
            print(f" - Output: {output}")
            
            #image URL from o/p
            image_url = output[0] if isinstance(output, list) else output
            print(f" - Image URL: {image_url}")
            
            #download image
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                image_b64 = base64.b64encode(image_response.content).decode('utf-8')
                print(f" Image downloaded and encoded successfully")
                return jsonify({
                    "image": image_b64,
                    "prompt": prompt,
                    "success": True
                })
            else:
                print(f" Failed to download image: {image_response.status_code}")
                return jsonify({"error": "Failed to download generated image"}), 500
        
        except Exception as e:
            print(f" Error in prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f" Error in generate_image: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=False, use_reloader=False, threaded=True)