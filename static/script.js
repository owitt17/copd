// Predefined questions and answers
const faq = {
    "What is COPD?": "COPD stands for Chronic Obstructive Pulmonary Disease, a group of lung diseases that block airflow and make breathing difficult.",
    "What are the symptoms of COPD?": "Common symptoms include shortness of breath, wheezing, chest tightness, and chronic cough.",
    "What causes COPD?": "The primary causes are smoking, long-term exposure to air pollution, chemical fumes, and dust.",
    "Can COPD be cured?": "COPD cannot be cured, but treatment can help manage symptoms and improve quality of life.",
    "What are the treatments for COPD?": "Treatments include medication, oxygen therapy, pulmonary rehabilitation, and in severe cases, surgery.",
    "Who is at risk for COPD?": "Smokers, people exposed to secondhand smoke, and those with a history of lung infections or occupational exposure to irritants are at higher risk.",
    "How is COPD diagnosed?": "COPD is diagnosed using spirometry, a lung function test, along with imaging tests and medical history evaluation.",
    "How can I prevent COPD?": "Avoid smoking, minimize exposure to lung irritants, maintain good air quality, and follow a healthy lifestyle to prevent COPD."
};

// Chatbot logic
const chatBody = document.getElementById("chat-body");
const questionButtons = document.getElementById("question-buttons");

function appendMessage(sender, message) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `${sender}-message`;
    
    const messageText = document.createElement("p");
    messageText.innerHTML = message;  
    
    messageDiv.appendChild(messageText);
    chatBody.appendChild(messageDiv);

    // Scroll to the latest message
    chatBody.scrollTop = chatBody.scrollHeight;
}

// Function to send a question to the chatbot API
function askChatbot(question) {
    fetch("/chatbot_api", {
        method: "POST",
        body: JSON.stringify({ message: question }),
        headers: {
            "Content-Type": "application/json"
        }
    })
    .then(response => response.json())
    .then(data => {
        appendMessage("bot", data.response);
    })
    .catch(error => {
        appendMessage("bot", "Sorry, I couldn't process your request.");
        console.error("Error:", error);
    });
}

// Generate question buttons dynamically
function generateQuestionButtons() {
    Object.keys(faq).forEach((question) => {
        const button = document.createElement("button");
        button.textContent = question;

        // Add click event for the button
        button.addEventListener("click", () => {
            appendMessage("user", question);
            setTimeout(() => {
                appendMessage("bot", faq[question]);
            }, 500);
        });

        // Add button to the question buttons container
        questionButtons.appendChild(button);
    });

    // Add a custom input field for user questions
    const inputContainer = document.createElement("div");
    inputContainer.className = "input-container";

    const inputField = document.createElement("input");
    inputField.type = "text";
    inputField.placeholder = "Ask me anything about COPD...";
    inputField.className = "chat-input";

    const sendButton = document.createElement("button");
    sendButton.textContent = "Send";
    sendButton.className = "send-btn";

    // Send question when clicking the button
    sendButton.addEventListener("click", () => {
        const question = inputField.value.trim();
        if (question) {
            appendMessage("user", question);
            inputField.value = ""; // Clear input field

            if (faq[question]) {
                // If question exists in predefined FAQ, use the predefined answer
                setTimeout(() => {
                    appendMessage("bot", faq[question]);
                }, 500);
            } else {
                // Otherwise, ask OpenAI API
                askChatbot(question);
            }
        }
    });

    // Allow sending question with Enter key
    inputField.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            sendButton.click();
        }
    });

    inputContainer.appendChild(inputField);
    inputContainer.appendChild(sendButton);
    questionButtons.appendChild(inputContainer);
}

// Initialize chatbot
generateQuestionButtons();
