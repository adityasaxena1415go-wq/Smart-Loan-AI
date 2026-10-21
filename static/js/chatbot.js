// ===========================
// SMARTLOAN AI CHATBOT
// ===========================

const chatBtn = document.getElementById("chatToggle");
const chatBox = document.getElementById("chatBox");
const messages = document.getElementById("chatMessages");
const input = document.getElementById("userMessage");

// Open / Close Chat
chatBtn.addEventListener("click", () => {

    if(chatBox.style.display === "flex"){
        chatBox.style.display = "none";
    }
    else{
        chatBox.style.display = "flex";
        input.focus();
    }

});

// Send on Enter
input.addEventListener("keypress", function(e){

    if(e.key === "Enter"){
        sendMessage();
    }

});

// Add Message Bubble
function addMessage(text, sender){

    let div = document.createElement("div");

    div.className = sender === "user"
        ? "user-message"
        : "bot-message";

    div.innerHTML = text.replace(/\n/g,"<br>");

    messages.appendChild(div);

    messages.scrollTop = messages.scrollHeight;

}

// Send Message
function sendMessage(){

    let msg = input.value.trim();

    if(msg==="") return;

    addMessage(msg,"user");

    input.value="";

    fetch("/chatbot",{

        method:"POST",

        headers:{
            "Content-Type":"application/x-www-form-urlencoded"
        },

        body:"message="+encodeURIComponent(msg)

    })

    .then(response=>response.json())

    .then(data=>{

        setTimeout(()=>{

            addMessage(data.reply,"bot");

        },500);

    })

    .catch(()=>{

        addMessage("⚠ Unable to connect to SmartLoan AI Assistant.","bot");

    });

}