# Cloud-Based Photography Agent Application

## 📌 Overview
This project is a cloud-based application where users upload images, add descriptions, and train a custom AI model (Flux-Dev) to generate outputs based on input prompts. It integrates user authentication, image handling, model training, and image generation into a seamless end-to-end workflow.

## 🚀 Features
- User registration and login (securely managed with bcrypt and sessions)
- Upload and manage 15 categorized images per project (person/product)
- Backend using Flask for image processing, model training, and API integration
- Uses Replicate API for AI training with LoRA fine-tuning
- Stores trained models on Hugging Face
- Sends email notifications via SendGrid upon training completion
- Streamlit frontend for user interaction, model selection, and image generation

## Live server link
 http://35.225.4.144:8501/

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Backend**: Flask
- **Database**: AWS MySQL
- **AI/ML**: Replicate API (Flux-Dev model), Hugging Face (model storage)
- **Email Service**: SendGrid
- **Authentication**: bcrypt, token-based sessions

## 🗂️ System Architecture
```
User → Streamlit (Frontend) → Flask (Backend) → MySQL (Database)
                                       ↓
                          Replicate API (Training) → Hugging Face (Model Storage)
                                       ↓
                             SendGrid (Email Notifications)
```

## System Block diagram

<img width="274" alt="image" src="https://github.com/user-attachments/assets/599d13d8-7f72-46be-b05f-7bbd1133f172" />

## 📁 Data Organization
- **Users Table**: Stores user credentials
- **Images Table**: Descriptions and categories linked by user_id
- **Projects Table**: Metadata, model type, training status

## 🔄 Flow
1. User registers/logs in
2. Uploads 15 categorized images with descriptions
3. Submits data for training
4. Model is trained using Replicate API, LoRA saved to Hugging Face
5. User receives an email upon completion
6. Selects trained model, provides prompt → Generates image

## 🔐 Security
- Password hashing via bcrypt
- Token-based session authentication
- API access control for all endpoints

## ⚠️ Limitations
- High training time and cloud computational costs
- May overfit with small datasets
- Dependence on external APIs (Replicate, SendGrid)
- Limited control over prompt effectiveness

## 🔮 Future Enhancements
- Support for video generation
- Social media sharing features
- AI-powered model recommendations
- Real-time collaboration for multi-user projects
- Multi-language prompts and interaction

## 👩‍💻 Author
**Snigdha Gogineni**  
Cloud-based AI Application | AI Image Generation | LoRA Fine-tuning




