import streamlit as st
import requests
import base64
from PIL import Image
from datetime import datetime
import io, json

st.set_page_config(layout="wide")

BACKEND_URL = "http://localhost:6000"
MAX_IMAGE_SIZE = (1024, 1024)  #better quality
PREVIEW_SIZE = (400, 400)  #preview size
JPEG_QUALITY = 100  #max quality

#process and resizing images for upload
def process_image(image_file, target_size=MAX_IMAGE_SIZE, quality=JPEG_QUALITY):
    
    try:
        if isinstance(image_file, bytes):
            image = Image.open(io.BytesIO(image_file))
        else:
            image = Image.open(image_file)
        
        if image.mode in ('RGBA', 'P'):
            image = image.convert('RGB')
        aspect_ratio = image.size[0] / image.size[1]
        if image.size[0] <= target_size[0] and image.size[1] <= target_size[1]:
            new_size = image.size
        else:
            new_size = (
                target_size[0], 
                int(target_size[0] / aspect_ratio)
            ) if aspect_ratio > 1 else (
                int(target_size[1] * aspect_ratio),
                target_size[1])
        
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        output_buffer = io.BytesIO()
        resized_image.save(output_buffer, format='JPEG', quality=quality, optimize=True)
        return output_buffer.getvalue()
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None
    
#backend server: running
def check_server_connection():
    try:
        response = requests.get(f"{BACKEND_URL}/", timeout=5)
        return True
    except:
        return False

#CSS for image display
st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
        padding: 16px;
    }

    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"]:first-child {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        position: sticky;
        top: 20px;
    }

    .element-container:has(h3) {
        margin-top: 1rem;
    }

    .uploadedImage {
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        margin: 8px 0;
    }

    .stTextArea {
        min-height: 120px;
    }
    
    .image-preview-container {
        padding: 12px;
        border: 1px solid #ddd;
        border-radius: 6px;
        background: #fafafa;
    }
    
    .image-preview {
        max-width: 400px;
        margin: 0 auto;
    }
    
    img {
        max-width: 400px;
        object-fit: contain;
        border-radius: 6px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
    }
    
    .image-caption {
        font-size: 14px;
        color: #555;
        margin-top: 8px;
        text-align: center;
    }
    
    hr {
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def display_image_section(files, category):
    #uploaded images with descriptions
    if not files:
        return
    
    #session state for descriptions if not done
    if f"desc_{category}_0" not in st.session_state:
        for i in range(5):
            st.session_state[f"desc_{category}_{i}"] = ""
    
    for i, file in enumerate(files):
        col1, col2 = st.columns([1, 2])
        with col1:
            image_bytes = file.read()
            st.image(image_bytes, caption=file.name, use_container_width=True)
            file.seek(0)
        
        with col2:
            description_key = f"desc_{category}_{i}"
            st.text_area(
                f"Description for {file.name}",
                key=description_key,
                value=st.session_state[description_key],
                on_change=update_description,
                args=(category, i)
            )
#description in session state
def update_description(category, index): 
    description_key = f"desc_{category}_{index}"
    st.session_state.descriptions[category][index] = st.session_state[description_key]

#initialize session state for descriptions
def init_session_state():  
    if "descriptions" not in st.session_state:
        st.session_state.descriptions = {
            #Person/Couple 
            "person1": ["" for _ in range(5)],
            "person2": ["" for _ in range(5)],
            "together": ["" for _ in range(5)],
            #Celebrity/Product 
            "celebrity": ["" for _ in range(5)],
            "product": ["" for _ in range(5)],
            "celeb_product": ["" for _ in range(5)]
        }
    
    #Initialize description fields if not present
    categories = ["person1", "person2", "together", "celebrity", "product", "celeb_product"]
    for category in categories:
        for i in range(5):
            key = f"desc_{category}_{i}"
            if key not in st.session_state:
                st.session_state[key] = ""

#start of model training
def handle_training_start(files1, files2, files3, model_type="person_couple", product_info=None): 

    #process all images
        all_files = []
        for i, file in enumerate(files1):
            processed = process_image(file)
            if processed:
                all_files.append({
                    "file": processed,
                    "description": st.session_state.descriptions["person1" if model_type == "person_couple" else "celebrity"][i],
                    "category": "person1" if model_type == "person_couple" else "celebrity"
                })
        
        for i, file in enumerate(files2):
            processed = process_image(file)
            if processed:
                all_files.append({
                    "file": processed,
                    "description": st.session_state.descriptions["person2" if model_type == "person_couple" else "product"][i],
                    "category": "person2" if model_type == "person_couple" else "product"
                })
        
        for i, file in enumerate(files3):
            processed = process_image(file)
            if processed:
                all_files.append({
                    "file": processed,
                    "description": st.session_state.descriptions["together" if model_type == "person_couple" else "celeb_product"][i],
                    "category": "together" if model_type == "person_couple" else "celeb_product"
                })
        
        if len(all_files) != 15:
            st.error("Failed to process all images. Please try again.")
            return
        
        #make form data
        form_data = {
            "username": st.session_state.username,
            "model_type": model_type
        }
        
        if model_type == "product_celebrity" and product_info:
            form_data["product_info"] = json.dumps(product_info)
        
        #prepare files
        files = []
        for i, file_data in enumerate(all_files):
            files.append(
                ("files", (f"{file_data['category']}_{i}.jpg", file_data["file"], "image/jpeg"))
            )
            form_data[f"desc_{i}"] = file_data["description"]
            form_data[f"category_{i}"] = file_data["category"]
        
        #send request to server
        response = requests.post(
            f"{BACKEND_URL}/upload_images",
            data=form_data,
            files=files
        )
        if response.status_code == 200:
            st.success("✅ Images uploaded successfully! Training will take approx 30 min.")
        else:
            st.error(f"Oops! Something went wrong while starting the training. Error: {response.json().get('error', 'Unknown error')}")
    
#display authentication
def auth_menu():
    st.title("Welcome to AI Image Platform")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        st.header("Login")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if email and password:
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/login",
                        json={"email": email, "password": password},
                        timeout=10  # Add timeout
                    )
                    
                    try:
                        response_data = response.json()
                        if response.status_code == 200:
                            st.session_state.logged_in = True
                            st.session_state.username = response_data["username"]
                            st.session_state.email = response_data["email"]
                            st.rerun()
                        else:
                            error_msg = response_data.get("error", "Login failed")
                            st.error(f"⚠️ {error_msg}")
                    except requests.exceptions.JSONDecodeError:
                        if response.status_code == 500:
                            st.error("⚠️ Server error. Please try again later.")
                        else:
                            st.error("⚠️ Unexpected response from server. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Could not connect to server. Please check if the server is running.")
                except requests.exceptions.Timeout:
                    st.error("⚠️ Server request timed out. Please try again.")
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
            else:
                st.error("Please fill in both email and password fields.")
    
    with tab2:
        st.header("Register")
        username = st.text_input("Username", key="reg_username")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_password")
        
        if st.button("Register"):
            if username and email and password:
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/register",
                        json={
                            "username": username,
                            "email": email,
                            "password": password
                        },
                        timeout=10  # Add timeout
                    )
                    
                    try:
                        response_data = response.json()
                        if response.status_code == 201:
                            st.success("Welcome! Your account is all set. Please log in to get started!")
                        else:
                            error_msg = response_data.get("error", "Registration failed")
                            st.error(f"⚠️ {error_msg}")
                    except requests.exceptions.JSONDecodeError:
                        if response.status_code == 500:
                            st.error("⚠️ Server error. Please try again later.")
                        else:
                            st.error("⚠️ Unexpected response from server. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Could not connect to server. Please check if the server is running.")
                except requests.exceptions.Timeout:
                    st.error("⚠️ Server request timed out. Please try again.")
                except Exception as e:
                    st.error(f"⚠️ An error occurred: {str(e)}")
            else:
                st.error("Please fill in all fields to create your account.")

def main():
    if not check_server_connection():
        st.error("""
         Cannot connect to the server. Please check:
         The Flask server is running (`python server_new.py`) and is accessible on port 5001.
        """)
        return

    #session state initialization
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.email = None

    if not st.session_state.logged_in:
        auth_menu()
    else:
        main_application()

def main_application():
    if st.button("Logout", key="logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.email = None
        st.rerun()
    
    #initialize session state
    init_session_state()
    
    #container for status messages
    status_container = st.empty()
    tab1, tab2 = st.tabs(["Train Model", "Generate Images"])
    with tab1:
        left_col, right_col = st.columns([1, 2])
        
        with left_col:
            st.header("Training Guidelines")
            
            #tabs for different model types
            model_type = st.radio(
                "Select Model Type",
                ["Person/Couple", "Celebrity/Product"],
                key="model_type",
                help="Choose the type of model you want to train"
            )
            
            if model_type == "Person/Couple":
                st.info("""
                **Image Requirements:**
                Upload 15 images total:
                - 5 pictures of Person 1
                - 5 pictures of Person 2
                - 5 pictures of both persons together
                """)
            else:
                st.info("""
                **Image Requirements:**
                Upload 15 images total:
                - 5 pictures of the Celebrity
                - 5 pictures of the Product
                - 5 pictures of Celebrity with Product
                """)
        
        with right_col:
            st.header("Upload Images & Descriptions")
            
            if model_type == "Person/Couple":
                #person 1
                st.subheader("Person 1 Images")
                person1_files = st.file_uploader("Upload 5 images of Person 1",type=["png", "jpg", "jpeg"],accept_multiple_files=True,key="person1_images")
                display_image_section(person1_files, "person1")
                st.markdown("---")
                #person 2
                st.subheader("Person 2 Images")
                person2_files = st.file_uploader("Upload 5 images of Person 2",type=["png", "jpg", "jpeg"],accept_multiple_files=True,key="person2_images")
                display_image_section(person2_files, "person2")
                st.markdown("---")
                st.subheader("Images of Both Persons")
                together_files = st.file_uploader("Upload 5 images of both persons together",type=["png", "jpg", "jpeg"],accept_multiple_files=True,key="together_images")
                display_image_section(together_files, "together")
                
                #checking requirements and start training
                all_files = (person1_files or []) + (person2_files or []) + (together_files or [])
                has_all_images = (len(person1_files or []) == 5 and 
                                len(person2_files or []) == 5 and 
                                len(together_files or []) == 5)
                
                if all_files:
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if not has_all_images:
                            st.warning("Please upload exactly 5 images for each category")
                        else:
                            if st.button(" Start Training", type="primary", use_container_width=True):
                                with st.spinner("Uploading images..."):
                                    handle_training_start(person1_files, person2_files, together_files)
                                    st.success("""
                                    You will receive an email once the training is finished.
                                    
                                    You can use the 'Generate Images' tab to create new images with your trained model/ can 'Generate images' from existing models.
                                    """)
            else:
                #celebrity/product UI
                st.subheader("Celebrity Images")
                celebrity_files = st.file_uploader("Upload 5 images of the Celebrity",type=["png", "jpg", "jpeg"],accept_multiple_files=True,
                                        key="celebrity_images")
                display_image_section(celebrity_files, "celebrity")
                st.markdown("---")
                #product details
                st.subheader("Product Details")
                product_name = st.text_input("Product Name", key="product_name",help="Enter the name of your product")
                product_category = st.selectbox("Product Category",
                                        ["Clothing", "Accessories", "Electronics", "Beauty", "Other"],key="product_category")
                product_style = st.text_area("Product Style Description",help="Describe the style/aesthetic of your product",key="product_style")
                st.subheader("Product Images")
                product_files = st.file_uploader("Upload 5 images of the Product",type=["png", "jpg", "jpeg"],accept_multiple_files=True,key="product_images")
                display_image_section(product_files, "product")
                
                st.markdown("---")
                
                st.subheader("Celebrity with Product")
                celeb_product_files = st.file_uploader("Upload 5 images of Celebrity with Product",
                                                   type=["png", "jpg", "jpeg"],accept_multiple_files=True,
                                                key="celeb_product_images")
                display_image_section(celeb_product_files, "celeb_product")
                
                #requirements and start training
                all_files = (celebrity_files or []) + (product_files or []) + (celeb_product_files or [])
                has_all_images = (len(celebrity_files or []) == 5 and 
                            len(product_files or []) == 5 and 
                            len(celeb_product_files or []) == 5)
                has_product_info = bool(product_name and product_category and product_style)
                
                if all_files:
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if not has_all_images:
                            st.warning("Please upload exactly 5 images for each category")
                        elif not has_product_info:
                            st.warning("Please fill in all product details")
                        else:
                            if st.button(" Start Training", type="primary", use_container_width=True):
                                product_info = {
                                    "product_name": product_name,
                                    "product_category": product_category,
                                    "product_style": product_style,
                                }
                                with st.spinner("Uploading images..."):
                                    handle_training_start(
                                        celebrity_files, product_files, celeb_product_files,
                                        model_type="product_celebrity",
                                        product_info=product_info
                                    )
                                    st.success("""
                                    You will receive an email once training is finished.
                                    
                                    You can use the 'Generate Images' tab to create new images with your trained model/ can 'Generate images' from existing models.
                                    """)
                            
    with tab2:
        st.header("Generate Images")
        
        try:
            with st.spinner("Loading available models..."):
              
                print(f"Fetching models for user: {st.session_state.username}")
                
                models_response = requests.get(
                    f"{BACKEND_URL}/get_trained_models",
                    params={
                        "username": st.session_state.username,
                        "model_type": "all"  #all model types
                    }
                )
                
                if models_response.status_code == 200:
                    models = models_response.json()
                    print(f"Received models: {models}") 
                    
                    if not models:
                        st.info("""
                        No trained models available yet. Train the model to generate images!
                        """)
                    else:
                        #models by type
                        person_models = [m for m in models if m['model_type'] == 'person_couple']
                        product_models = [m for m in models if m['model_type'] == 'product_celebrity']
                        
                        #different model types selection
                        model_type = st.radio(
                            "Select Model Type",
                            options=["Person/Couple Models", "Product/Celebrity Models"],
                            index=0 if person_models else 1,
                            key="generate_model_type"
                        )
                        
                        #display names to actual model types
                        current_models = person_models if model_type == "Person/Couple Models" else product_models
                        
                        if current_models:
                            selected_model = st.selectbox(
                                "Select Model",
                                options=current_models,
                                format_func=lambda x: f"{x['name']} (Created: {x['created_at'][:10]})",
                                key="selected_model"
                            )
                            
                            if selected_model:
                                #model specific information
                                with st.expander("Model Details", expanded=False):
                                    st.write(f"Model Name: {selected_model['name']}")
                                    st.write(f"Created: {selected_model['created_at']}")
                                    st.write(f"Model ID: {selected_model['model_id']}")
                                    if selected_model['model_type'] == 'person_couple':
                                        st.write(f"Trigger Word: {selected_model['trigger_word']}")
                                    else:
                                        if 'product_info' in selected_model:
                                            st.write("Product Information:")
                                            st.json(selected_model['product_info'])
                                
                                #prompt input with help 
                                prompt_help = """
                                Describe the image you want to generate. Tips:
                                - Be specific about the scene, setting
                                - Be specific about endorsing the product
                                - **IMPORTANT**: Include the trigger word "{}" in your prompt
                                """.format(selected_model.get('trigger_word', 'wxy_mno'))
                                
                                #trigger word
                                trigger_word = selected_model.get('trigger_word', '')
                                 #example prompt with trigger word
                                example_prompt = f"a portrait of beautiful couple/celebrity {trigger_word} in a beautiful garden, natural lighting"
                                st.info(f"Example prompt: `{example_prompt}`")
                                prompt = st.text_area("Enter your prompt:", help=prompt_help)
                                
                                if st.button("Generate Image"):
                                    if not prompt:
                                        st.error("Please enter a prompt.")
                                    else:
                                        #trigger word is in prompt
                                        if trigger_word and trigger_word not in prompt:
                                            prompt = f"{trigger_word}, {prompt}"
                                        
                                        print(f"\n=== Generating image ===")
                                        print(f"  - Model: {selected_model['model_id']}")
                                        print(f"  - Trigger word: {trigger_word}")
                                        print(f"  - Prompt: {prompt}")
                                        
                                        with st.spinner("Creating your image... This may take a minute..."):
                                            try:
                                                response = requests.post(
                                                    f"{BACKEND_URL}/generate_image",
                                                    json={
                                                        "model_id": selected_model["model_id"],
                                                        "username": st.session_state.username,
                                                        "prompt": prompt
                                                    },
                                                    timeout=300  # 5 min timeout
                                                )
                                                
                                                if response.status_code == 200:
                                                    data = response.json()
                                                    image_bytes = base64.b64decode(data["image"])
                                                    image = Image.open(io.BytesIO(image_bytes))
                                                    
                                                    #display image with prompt
                                                    st.image(image, caption=data.get("prompt", prompt), use_container_width=True)
                                                    st.success("Image generated successfully!")
                                                    
                                                    #download button
                                                    buf = io.BytesIO()
                                                    image.save(buf, format='PNG')
                                                    st.download_button(
                                                        label="Download Image",
                                                        data=buf.getvalue(),
                                                        file_name=f"generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                                        mime="image/png"
                                                    )
                                                else:
                                                    error_msg = response.json().get("error", "Generation failed")
                                                    st.error(f"Failed to generate image: {error_msg}")
                                                    st.info("Please try a different prompt or model.")
                                            except Exception as e:
                                                st.error(f"An error occurred: {str(e)}")
                                                st.info("Please try again with a different prompt or model.")
                        else:
                            st.info(f"No {'person/couple' if model_type == 'Person/Couple Models' else 'product/celebrity'} models available yet.")
                else:
                    error_msg = models_response.json().get('error', 'Unknown error')
                    st.error(f"Error loading models: {error_msg}")
                    if "User not found" in error_msg:
                        st.warning("Please log out and log back in to refresh your session.")
        except requests.exceptions.RequestException as e:
            st.info("Please check if the server is running and try again.")
        except Exception as e:
            st.info("Please try refreshing the page.")

if __name__ == "__main__":
    main()