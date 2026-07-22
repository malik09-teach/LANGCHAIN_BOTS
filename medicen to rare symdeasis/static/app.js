document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('chatForm');
    const input = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const sessionId = generateSessionId();

    function generateSessionId() {
        return Math.random().toString(36).substring(2, 15);
    }

    function addMessage(content, type, isHtml = false) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (isHtml) {
            contentDiv.innerHTML = content;
        } else {
            contentDiv.textContent = content;
        }
        
        msgDiv.appendChild(contentDiv);
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function addLoadingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message system loading-msg';
        msgDiv.id = 'loadingIndicator';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = `
            <div class="loading-indicator">
                <span></span><span></span><span></span>
            </div>
        `;
        
        msgDiv.appendChild(contentDiv);
        chatContainer.appendChild(msgDiv);
        scrollToBottom();
    }

    function removeLoadingIndicator() {
        const indicator = document.getElementById('loadingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }

    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const text = input.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        input.value = '';
        input.disabled = true;
        
        addLoadingIndicator();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    session_id: sessionId,
                    message: text
                })
            });

            const data = await response.json();
            removeLoadingIndicator();
            addMessage(data.response, 'system', true);
        } catch (error) {
            console.error('Error:', error);
            removeLoadingIndicator();
            addMessage('An error occurred while processing your request.', 'system');
        } finally {
            input.disabled = false;
            input.focus();
        }
    });
});
