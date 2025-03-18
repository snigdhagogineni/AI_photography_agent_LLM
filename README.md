# AI_photography_agent

 live server link- http://35.225.4.144:8501/

My approach for developing this application:
1. User Authentication: Users can sign up and log in using username, email and password.
2. Image Upload & Management: Users should upload 15 images along with descriptions for each image.
3. Backend Processing:
   - Assigned code names for each person (e.g., 'wxy') to maintain user abstraction.
   - Stores images securely in a cloud server under user’s account(MySQL).
4. AI Model Training:
   - Utilized the Flux-Dev model(ostris/flux-dev-lora), to train via Replicate API.
   - Saves trained LoRA (Low-Rank Adaptation) file in a private Hugging Face repository.
5. Autotomated Notifications: Sends email when training completes using SendGrid.
6. Interactive Interface: Users select a trained LoRA model and input text prompts to generate AI-images
7. Product Marketing: AI-based marketing by training on celebrity and product images.

System Block diagram-
<img width="274" alt="image" src="https://github.com/user-attachments/assets/599d13d8-7f72-46be-b05f-7bbd1133f172" />


