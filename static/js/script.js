document.addEventListener('DOMContentLoaded', () => {
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const messagesDiv = document.getElementById('messages');

    chatForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage(query, 'user');
        userInput.value = '';

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong');
            }

            const result = await response.json();
            console.log(result); // For debugging: see the actual API response structure
            
            // Robustly select the main answer/summary
            let mainText = '';
            if (typeof result.summary === 'string' && result.summary.trim() !== '') {
                mainText = result.summary;
            } else if (result.results && typeof result.results.summary === 'string' && result.results.summary.trim() !== '') {
                mainText = result.results.summary;
            } else if (typeof result.answer === 'string' && result.answer.trim() !== '') {
                mainText = result.answer;
            } else if (typeof result.response === 'string' && result.response.trim() !== '') {
                mainText = result.response;
            } else {
                mainText = 'No summary or answer available.';
            }
            appendMessage(mainText, 'bot');
            
            // Display sources if available (especially for RAG responses)
            if (result.sources && result.sources.length > 0) {
                appendSources(result.sources, result.analysis_type);
            }
            
            // Display retrieved documents for RAG responses
            if (result.retrieved_documents && result.retrieved_documents.length > 0) {
                appendRetrievedDocuments(result.retrieved_documents);
            }
            
            // Handle visualizations if available
            if (result.visualizations && result.visualizations.length > 0) {
                appendVisualizations(result.visualizations);
            }

        } catch (error) {
            console.error('Error:', error);
            appendMessage(`Error: ${error.message}`, 'bot');
        }
    });

    // Simple Markdown to HTML converter (for bolding)
    const markdownToHtml = (text) =>
        (typeof text === 'string' ? text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') : '');

    function appendMessage(text, sender) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);
        
        const bubbleElement = document.createElement('div');
        bubbleElement.classList.add('message-bubble');
        
        // Apply markdown to HTML conversion only for bot messages
        if (sender === 'bot') {
            bubbleElement.innerHTML = markdownToHtml(text || '');
        } else {
            bubbleElement.innerHTML = text || '';
        }

        messageElement.appendChild(bubbleElement);
        messagesDiv.appendChild(messageElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function appendSources(sources, analysisType) {
        const sourcesElement = document.createElement('div');
        sourcesElement.classList.add('message', 'bot');
        
        const bubbleElement = document.createElement('div');
        bubbleElement.classList.add('message-bubble', 'sources-bubble');
        
        const sourcesTitle = document.createElement('div');
        sourcesTitle.style.fontWeight = 'bold';
        sourcesTitle.style.marginBottom = '8px';
        sourcesTitle.textContent = analysisType === 'conversational' ? '📚 Sources Used:' : '📊 Data Sources:';
        
        bubbleElement.appendChild(sourcesTitle);
        
        const sourcesList = document.createElement('ul');
        sourcesList.style.margin = '0';
        sourcesList.style.paddingLeft = '20px';
        
        sources.forEach(source => {
            const sourceItem = document.createElement('li');
            sourceItem.textContent = source;
            sourceItem.style.marginBottom = '4px';
            sourcesList.appendChild(sourceItem);
        });
        
        bubbleElement.appendChild(sourcesList);
        sourcesElement.appendChild(bubbleElement);
        messagesDiv.appendChild(sourcesElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function appendRetrievedDocuments(documents) {
        const docsElement = document.createElement('div');
        docsElement.classList.add('message', 'bot');
        
        const bubbleElement = document.createElement('div');
        bubbleElement.classList.add('message-bubble', 'docs-bubble');
        
        const docsTitle = document.createElement('div');
        docsTitle.style.fontWeight = 'bold';
        docsTitle.style.marginBottom = '8px';
        docsTitle.textContent = '🔍 Retrieved Context:';
        
        bubbleElement.appendChild(docsTitle);
        
        documents.slice(0, 3).forEach((doc, index) => {
            const docElement = document.createElement('div');
            docElement.style.marginBottom = '8px';
            docElement.style.padding = '8px';
            docElement.style.backgroundColor = 'rgba(255, 255, 255, 0.1)';
            docElement.style.borderRadius = '4px';
            docElement.style.fontSize = '0.9em';
            
            const docContent = document.createElement('div');
            docContent.textContent = doc.content.substring(0, 150) + (doc.content.length > 150 ? '...' : '');
            docElement.appendChild(docContent);
            
            if (doc.metadata && Object.keys(doc.metadata).length > 0) {
                const metadataElement = document.createElement('div');
                metadataElement.style.fontSize = '0.8em';
                metadataElement.style.opacity = '0.7';
                metadataElement.style.marginTop = '4px';
                metadataElement.textContent = `Year: ${doc.metadata.year || 'N/A'}, State: ${doc.metadata.state || 'N/A'}`;
                docElement.appendChild(metadataElement);
            }
            
            bubbleElement.appendChild(docElement);
        });
        
        docsElement.appendChild(bubbleElement);
        messagesDiv.appendChild(docsElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    function appendVisualizations(visualizations) {
        const vizElement = document.createElement('div');
        vizElement.classList.add('message', 'bot');
        
        const bubbleElement = document.createElement('div');
        bubbleElement.classList.add('message-bubble', 'viz-bubble');
        
        const vizTitle = document.createElement('div');
        vizTitle.style.fontWeight = 'bold';
        vizTitle.style.marginBottom = '8px';
        vizTitle.textContent = '📊 Visualizations:';
        
        bubbleElement.appendChild(vizTitle);
        
        visualizations.forEach((plotData, index) => {
            const plotDiv = document.createElement('div');
            plotDiv.id = `plotly-chart-${Date.now()}-${index}`;
            plotDiv.style.width = '100%';
            plotDiv.style.height = '400px';
            bubbleElement.appendChild(plotDiv);
            Plotly.newPlot(plotDiv.id, plotData.data, plotData.layout);
        });

        vizElement.appendChild(bubbleElement);
        messagesDiv.appendChild(vizElement);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }

    // Function to fetch and render quick stats
    async function loadQuickStats() {
        try {
            const response = await fetch('/api/quick_stats');
            if (!response.ok) {
                throw new Error('Failed to load quick stats');
            }
            const stats = await response.json();
            const quickStatsDiv = document.getElementById('quick-stats');
            
            quickStatsDiv.innerHTML = `
                <div class="stat-item">
                    <div class="value">${stats.total_records.toLocaleString()}</div>
                    <div class="label">Total Records</div>
                </div>
                <div class="stat-item">
                    <div class="value">${stats.years_covered}</div>
                    <div class="label">Years Covered</div>
                </div>
                <div class="stat-item">
                    <div class="value">${stats.states_uts}</div>
                    <div class="label">States/UTs</div>
                </div>
                <div class="stat-item">
                    <div class="value">${stats.crime_types}</div>
                    <div class="label">Crime Types</div>
                </div>
            `;
        } catch (error) {
            console.error('Error loading quick stats:', error);
            document.getElementById('quick-stats').innerHTML = '<p>Error loading stats.</p>';
        }
    }

    // Function to fetch and render filter options
    async function loadFilters() {
        try {
            const response = await fetch('/api/filter_options');
            if (!response.ok) {
                throw new Error('Failed to load filter options');
            }
            const filterOptions = await response.json();
            const filtersDiv = document.getElementById('filters');

            // Helper function to create checkbox group HTML
            const createCheckboxGroup = (id, labelText, options) => {
                return `
                    <div class="filter-group">
                        <label>${labelText}:</label>
                        <div class="filter-checkbox-container" id="${id}-checkboxes">
                            ${options.map(option => `
                                <div class="checkbox-item">
                                    <input type="checkbox" id="${id}-${option}" value="${option}">
                                    <label for="${id}-${option}">${option}</label>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            };

            filtersDiv.innerHTML = `
                ${createCheckboxGroup('year-filter', 'Year', filterOptions.years)}
                ${createCheckboxGroup('state-filter', 'State/UT', filterOptions.states)}
                ${createCheckboxGroup('crime-type-filter', 'Crime Type', filterOptions.crime_types)}
                <button id="generate-prompt-button" class="sample-query-button" style="width: 100%;">Generate Prompt 📝</button>
            `;

            document.getElementById('generate-prompt-button').addEventListener('click', applyFilters);

        } catch (error) {
            console.error('Error loading filters:', error);
            document.getElementById('filters').innerHTML = '<p>Error loading filters.</p>';
        }
    }

    // Function to apply filters and generate prompt
    async function applyFilters() {
        const getSelectedCheckboxes = (containerId) => {
            const container = document.getElementById(containerId);
            if (!container) return [];
            return Array.from(container.querySelectorAll('input[type="checkbox"]:checked')).map(checkbox => checkbox.value);
        };

        const selectedYears = getSelectedCheckboxes('year-filter-checkboxes');
        const selectedStates = getSelectedCheckboxes('state-filter-checkboxes');
        const selectedCrimeTypes = getSelectedCheckboxes('crime-type-filter-checkboxes');

        let queryParts = [];
        if (selectedYears.length > 0) {
            queryParts.push(`years ${selectedYears.join(', ')}`);
        }
        if (selectedStates.length > 0) {
            queryParts.push(`states ${selectedStates.join(', ')}`);
        }
        if (selectedCrimeTypes.length > 0) {
            queryParts.push(`crime types ${selectedCrimeTypes.join(', ')}`);
        }

        let generatedQuery = "";
        if (queryParts.length > 0) {
            generatedQuery = `Show me statistics for ${queryParts.join(', and ')}.`;
        } else {
            generatedQuery = "Show me overall crime statistics.";
        }

        // Set the generated query to the main input field and trigger submission
        userInput.value = generatedQuery;
        // Directly trigger submission by dispatching a submit event on the form
        chatForm.dispatchEvent(new Event('submit'));
    }

    // Function to fetch and render sample queries
    async function loadSampleQueries() {
        try {
            const response = await fetch('/api/sample_queries');
            if (!response.ok) {
                throw new Error('Failed to load sample queries');
            }
            const categories = await response.json();
            const sampleQueriesDiv = document.getElementById('sample-queries');
            sampleQueriesDiv.innerHTML = ''; // Clear existing content

            for (const category in categories) {
                if (categories.hasOwnProperty(category)) {
                    const categoryHeader = document.createElement('h4');
                    categoryHeader.textContent = category;
                    sampleQueriesDiv.appendChild(categoryHeader);

                    categories[category].forEach(queryText => {
                        const button = document.createElement('button');
                        button.classList.add('sample-query-button');
                        button.textContent = queryText;
                        button.addEventListener('click', () => {
                            userInput.value = queryText;
                            chatForm.dispatchEvent(new Event('submit')); // Programmatically submit the form
                        });
                        sampleQueriesDiv.appendChild(button);
                    });
                }
            }

        } catch (error) {
            console.error('Error loading sample queries:', error);
            document.getElementById('sample-queries').innerHTML = '<p>Error loading sample queries.</p>';
        }
    }

    // Call all load functions when the page loads
    loadQuickStats();
    loadFilters();
    loadSampleQueries();
}); 