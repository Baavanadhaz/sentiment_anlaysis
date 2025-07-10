E-commerce Sentiment Analysis
This project predicts customer sentiment (Positive or Negative) based purely on e-commerce transaction details such as:

Product Category

Payment Method

Quantity

Price

Built using Python, Flask, Scikit-learn, HTML/CSS, and deployed as a web-based interface.

Technologies Used

Tool/Library	Purpose
Python    	   Programming Language
Flask          Web Framework for backend
HTML,CSS,JS	   Front-end interface
Pandas	       Data preprocessing
Scikit-learn	 Machine Learning (Random Forest)
Joblib	       Model serialization and loading

Project Objective
To classify customer sentiment as Positive or Negative based only on transaction behavior — without relying on written reviews.

This enables automated sentiment estimation based on purchase patterns.

🔍 How It Works
The model is trained on historical e-commerce data labeled using the ReviewScore:

4–5 → Positive

1–2 → Negative

3 → Ignored (neutral)

Features used:

Category (e.g., Books, Electronics)

Payment Method (e.g., Credit Card, PayPal)

Quantity

Price (in USD)

The model is trained using a Random Forest Classifier, with categorical features one-hot encoded and numerical features scaled.

🌐 Web Interface
The Flask-based web app lets users:

Choose category and payment method

Enter quantity and price

Click Predict to get a sentiment result

The model processes the input and returns:

✅ Positive or ❌ Negative

▶️ How to Run the Project
Step 1: Install dependencies
bash
Copy
Edit
pip install flask pandas scikit-learn joblib
Step 2: Train the model
bash
Copy
Edit
python main.py
Step 3: Run the web app
bash
Copy
Edit
python app.py
Step 4: Open in browser
Navigate to: http://localhost:5000

🧪 Sample Use Case
Category: Clothing

Payment: PayPal

Quantity: 4

Price: 400

→ The model may predict: Positive ✅

📌 Future Improvements
Add more features (delivery time, location, etc.)

Store predictions in a database

Deploy to the cloud (Render, Heroku, etc.)

Add data visualizations for admin view

👤 Author
Baavana M
Final Year IT Student | Passionate about AI, Automation, and Web Apps 

