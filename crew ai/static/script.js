document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const statusIndicator = document.getElementById('status-indicator');
    const terminal = document.getElementById('terminal');
    const resultBox = document.getElementById('result');
    
    let pollingInterval = null;
    
    startBtn.addEventListener('click', async () => {
        // Update UI
        startBtn.disabled = true;
        startBtn.innerText = 'Workflow Running...';
        
        statusIndicator.className = 'status running';
        statusIndicator.innerText = 'Status: Running';
        
        terminal.innerHTML = '';
        resultBox.innerHTML = '<span class="placeholder-text">Waiting for final output...</span>';
        
        try {
            // Trigger the backend to start
            await fetch('/api/run', { method: 'POST' });
            
            // Start polling logs every 1 second
            if (pollingInterval) clearInterval(pollingInterval);
            pollingInterval = setInterval(fetchLogs, 1000);
            
        } catch (error) {
            console.error('Error starting agent:', error);
            statusIndicator.className = 'status idle';
            statusIndicator.innerText = 'Status: Error';
            startBtn.disabled = false;
            startBtn.innerText = 'Start Agent Workflow';
        }
    });
    
    async function fetchLogs() {
        try {
            const response = await fetch('/api/logs');
            const data = await response.json();
            
            // Update Terminal
            if (data.updates) {
                terminal.innerText = data.updates;
                // Auto scroll to bottom
                terminal.scrollTop = terminal.scrollHeight;
            }
            
            // Check for completion
            if (data.updates && data.updates.includes('Agent Finished.')) {
                clearInterval(pollingInterval);
                statusIndicator.className = 'status completed';
                statusIndicator.innerText = 'Status: Completed';
                startBtn.disabled = false;
                startBtn.innerText = 'Start Again';
                
                if (data.final_post) {
                    // Render markdown using Marked.js (imported in HTML)
                    resultBox.innerHTML = marked.parse(data.final_post);
                } else {
                    resultBox.innerHTML = '<span class="placeholder-text">Agent finished but no final post found.</span>';
                }
            }
        } catch (error) {
            console.error('Error fetching logs:', error);
        }
    }
});
