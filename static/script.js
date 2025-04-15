document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictionForm');
    const steps = document.querySelectorAll('.form-step');
    const stepIndicators = document.querySelectorAll('.step');
    const progressBar = document.querySelector('.progress-bar');
    const prevButton = document.querySelector('.prev-step');
    const nextButton = document.querySelector('.next-step');
    const submitButton = document.querySelector('.submit-form');
    const loadingSpinner = document.getElementById('loading');
    const resultsCard = document.getElementById('results');
    const errorAlert = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');

    let currentStep = 1;
    const totalSteps = steps.length;

    // Configure marked options for better rendering
    marked.setOptions({
        breaks: true,
        gfm: true,
        headerIds: false,
        mangle: false
    });

    // Update progress bar and step indicators
    function updateProgress() {
        const progress = ((currentStep - 1) / (totalSteps - 1)) * 100;
        progressBar.style.width = `${progress}%`;

        stepIndicators.forEach((step, index) => {
            const stepNum = index + 1;
            step.classList.remove('active', 'completed');
            
            if (stepNum === currentStep) {
                step.classList.add('active');
            } else if (stepNum < currentStep) {
                step.classList.add('completed');
            }
        });

        // Update button visibility
        prevButton.style.display = currentStep === 1 ? 'none' : 'block';
        nextButton.style.display = currentStep === totalSteps ? 'none' : 'block';
        submitButton.style.display = currentStep === totalSteps ? 'block' : 'none';
    }

    // Validate current step
    function validateStep(step) {
        const currentStepElement = document.querySelector(`.form-step[data-step="${step}"]`);
        const requiredFields = currentStepElement.querySelectorAll('[required]');
        
        for (const field of requiredFields) {
            if (!field.value) {
                field.focus();
                return false;
            }
        }
        return true;
    }

    // Handle step navigation
    function goToStep(step) {
        if (step < 1 || step > totalSteps) return;
        
        if (step > currentStep && !validateStep(currentStep)) {
            showError('Please fill in all required fields before proceeding.');
            return;
        }

        steps.forEach(s => {
            s.classList.remove('active');
            s.style.display = 'none';
        });

        const targetStep = document.querySelector(`.form-step[data-step="${step}"]`);
        targetStep.style.display = 'block';
        
        setTimeout(() => {
            targetStep.classList.add('active');
        }, 50);

        currentStep = step;
        updateProgress();
    }

    // Event listeners for navigation
    prevButton.addEventListener('click', () => goToStep(currentStep - 1));
    nextButton.addEventListener('click', () => goToStep(currentStep + 1));

    // Step indicator clicks
    stepIndicators.forEach((indicator, index) => {
        indicator.addEventListener('click', () => {
            const targetStep = index + 1;
            if (targetStep < currentStep || validateStep(currentStep)) {
                goToStep(targetStep);
            }
        });
    });

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorAlert.classList.remove('d-none');
        errorAlert.style.opacity = '0';
        
        requestAnimationFrame(() => {
            errorAlert.style.opacity = '1';
        });

        setTimeout(() => {
            errorAlert.style.opacity = '0';
            setTimeout(() => {
                errorAlert.classList.add('d-none');
            }, 300);
        }, 3000);
    }

    // Format prediction results
    function formatPredictions(predictions) {
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.innerHTML = '';

        // Create data for the pie chart
        const chartData = {};
        predictions.forEach(prediction => {
            chartData[prediction.deficiency + ' Deficiency'] = prediction.probability / 100;
        });
        
        // Create all charts
        createPredictionChart(chartData);
        createSymptomCorrelationChart(predictions);
        createImpactAreasChart(predictions);
        createRiskLevelChart(predictions);

        predictions.forEach((prediction, index) => {
            const card = document.createElement('div');
            card.className = `prediction-card animate__animated animate__fadeInUp ${index === 0 ? 'primary-prediction' : ''}`;
            card.style.animationDelay = `${index * 0.1}s`;

            // Get appropriate icon based on deficiency type
            const icon = getPredictionIcon(prediction.deficiency);
            
            card.innerHTML = `
                <div class="prediction-header">
                    <div class="prediction-rank">
                        ${index + 1}
                    </div>
                    <div class="prediction-info">
                        <div class="prediction-name">
                            <i class="${icon} me-2"></i>
                            ${prediction.deficiency} Deficiency
                        </div>
                    </div>
                </div>
                <div class="prediction-probability">
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" 
                             style="width: 0%" 
                             aria-valuenow="${prediction.probability}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                    <div class="probability-value">
                        <span class="value">0</span>%
                    </div>
                </div>
            `;

            predictionResult.appendChild(card);

            // Animate the progress bar and percentage after a short delay
            setTimeout(() => {
                const progressBar = card.querySelector('.progress-bar');
                const valueDisplay = card.querySelector('.probability-value .value');
                const targetValue = prediction.probability;
                let currentValue = 0;

                const animateValue = () => {
                    if (currentValue < targetValue) {
                        currentValue = Math.min(currentValue + 1, targetValue);
                        valueDisplay.textContent = currentValue.toFixed(1);
                        progressBar.style.width = `${currentValue}%`;
                        requestAnimationFrame(animateValue);
                    }
                };

                animateValue();
            }, index * 200 + 300);
        });
    }

    // Create symptom correlation chart
    function createSymptomCorrelationChart(predictions) {
        const ctx = document.getElementById('symptomCorrelationChart').getContext('2d');
        
        // Get top 3 predictions
        const topPredictions = predictions.slice(0, 3);
        
        // Define common symptoms for each deficiency
        const commonSymptoms = {
            'Vitamin D': ['Fatigue', 'Bone Pain', 'Muscle Weakness', 'Depression'],
            'Vitamin B12': ['Fatigue', 'Memory Loss', 'Tingling', 'Weakness'],
            'Iron': ['Fatigue', 'Shortness of Breath', 'Pale Skin', 'Weakness'],
            'Vitamin C': ['Bleeding Gums', 'Slow Healing', 'Fatigue', 'Joint Pain'],
            'Vitamin B6': ['Skin Issues', 'Depression', 'Confusion', 'Fatigue'],
            'Folate': ['Fatigue', 'Weakness', 'Shortness of Breath', 'Headache']
        };

        const datasets = topPredictions.map((pred, index) => {
            const deficiency = pred.deficiency;
            const symptoms = commonSymptoms[deficiency] || [];
            const correlationValues = symptoms.map(() => (Math.random() * 0.4 + 0.6) * pred.probability); // Simulated correlation

            return {
                label: deficiency,
                data: correlationValues,
                backgroundColor: `rgba(${index === 0 ? '74, 144, 226' : index === 1 ? '46, 204, 113' : '231, 76, 60'}, 0.2)`,
                borderColor: index === 0 ? '#4a90e2' : index === 1 ? '#2ecc71' : '#e74c3c',
                borderWidth: 2
            };
        });

        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Symptom 1', 'Symptom 2', 'Symptom 3', 'Symptom 4'],
                datasets: datasets
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Correlation Strength (%)'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Create impact areas chart
    function createImpactAreasChart(predictions) {
        const ctx = document.getElementById('impactAreasChart').getContext('2d');
        
        // Get the highest probability prediction
        const topPrediction = predictions[0];
        
        // Define impact areas and their values based on the deficiency
        const impactAreas = [
            'Physical Health',
            'Mental Health',
            'Energy Levels',
            'Immune System',
            'Recovery'
        ];

        // Generate impact values based on the prediction probability
        const baseImpact = topPrediction.probability / 100;
        const impactValues = impactAreas.map(() => 
            baseImpact * (0.8 + Math.random() * 0.4) * 100 // Varied impact values
        );

        new Chart(ctx, {
            type: 'radar',
            data: {
                labels: impactAreas,
                datasets: [{
                    label: `${topPrediction.deficiency} Impact`,
                    data: impactValues,
                    backgroundColor: 'rgba(74, 144, 226, 0.2)',
                    borderColor: '#4a90e2',
                    borderWidth: 2,
                    pointBackgroundColor: '#4a90e2'
                }]
            },
            options: {
                responsive: true,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            stepSize: 20
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    // Create risk level chart
    function createRiskLevelChart(predictions) {
        const ctx = document.getElementById('riskLevelChart').getContext('2d');
        
        // Calculate risk levels based on prediction probabilities
        const totalRisk = predictions.reduce((sum, pred) => sum + pred.probability, 0) / 100;
        const riskLevels = {
            'High Risk': Math.min(totalRisk, 0.4) * 100,
            'Moderate Risk': Math.min(Math.max(totalRisk - 0.4, 0), 0.3) * 100,
            'Low Risk': Math.max(100 - (totalRisk * 100), 0)
        };

        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: Object.keys(riskLevels),
                datasets: [{
                    data: Object.values(riskLevels),
                    backgroundColor: [
                        '#e74c3c', // High Risk - Red
                        '#f1c40f', // Moderate Risk - Yellow
                        '#2ecc71'  // Low Risk - Green
                    ]
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                return `${label}: ${value.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Helper function to get appropriate icon for each deficiency type
    function getPredictionIcon(deficiency) {
        const iconMap = {
            'Iron': 'fas fa-fire',
            'Vitamin A': 'fas fa-eye',
            'Vitamin B1': 'fas fa-brain',
            'Vitamin B12': 'fas fa-dna',
            'Vitamin B2': 'fas fa-bolt',
            'Vitamin B3': 'fas fa-sun',
            'Vitamin B6': 'fas fa-heartbeat',
            'Vitamin C': 'fas fa-lemon',
            'Vitamin D': 'fas fa-sun',
            'Vitamin E': 'fas fa-shield-alt',
            'Vitamin K': 'fas fa-tint',
            'Folate': 'fas fa-leaf',
            'Zinc': 'fas fa-atom'
        };

        // Extract the base vitamin name from the deficiency string
        const baseName = deficiency.split(' ')[0] + (deficiency.split(' ')[1] || '');
        return iconMap[baseName] || 'fas fa-pills';
    }

    function formatDietRecommendation(recommendation) {
        console.log('Processing diet recommendation:', recommendation);
        
        const dietSection = document.createElement('div');
        dietSection.className = 'diet-recommendation animate__animated animate__fadeIn';
        
        if (!recommendation || recommendation.trim() === '') {
            console.log('No diet recommendation provided');
            dietSection.innerHTML = `
                <div class="diet-header">
                    <h2><i class="fas fa-utensils me-2"></i>Personalized Diet Recommendations</h2>
                </div>
                <div class="diet-content">
                    <p>No diet recommendations available at this time.</p>
                </div>
            `;
            return dietSection;
        }

        try {
            // Clean up the markdown text
            const cleanMarkdown = recommendation
                .replace(/\r\n/g, '\n')
                .replace(/\n\n+/g, '\n\n')
                .trim();
            
            console.log('Cleaned markdown:', cleanMarkdown);

            // Parse the markdown sections
            const sections = cleanMarkdown.split('\n\n');
            let formattedContent = '';
            let currentSection = '';

            sections.forEach((section, index) => {
                if (section.startsWith('##')) {
                    // Main header
                    const title = section.replace('##', '').trim();
                    formattedContent += `
                        <div class="diet-header">
                            <h2><i class="fas fa-utensils me-2"></i>${title}</h2>
                        </div>
                    `;
                } else if (section.startsWith('#')) {
                    // Section header
                    const title = section.replace('#', '').trim();
                    if (currentSection) {
                        formattedContent += `</div>`; // Close previous section
                    }
                    currentSection = title;
                    formattedContent += `
                        <div class="diet-section">
                            <h3>${title}</h3>
                    `;
                } else if (section.includes('* ')) {
                    // List items
                    const items = section
                        .split('\n')
                        .filter(item => item.trim().startsWith('*'))
                        .map(item => item.replace('*', '').trim())
                        .map(item => `
                            <li>
                                <div class="tip-item">
                                    <span class="tip-icon"><i class="fas fa-check-circle"></i></span>
                                    <div class="tip-content">${item}</div>
                                </div>
                            </li>
                        `)
                        .join('');
                    
                    formattedContent += `<ul class="diet-list">${items}</ul>`;
                } else {
                    // Regular paragraph
                    formattedContent += `
                        <div class="diet-card">
                            <p>${section}</p>
                        </div>
                    `;
                }
            });

            // Close the last section if open
            if (currentSection) {
                formattedContent += '</div>';
            }

            // Add tips section if it exists
            if (cleanMarkdown.includes('Tips for Better Absorption')) {
                const tipsSection = sections
                    .find(section => section.includes('Tips for Better Absorption'));
                
                if (tipsSection) {
                    const tips = tipsSection
                        .split('\n')
                        .filter(line => line.trim().startsWith('*'))
                        .map(tip => tip.replace('*', '').trim())
                        .map(tip => `
                            <div class="tip-item">
                                <span class="tip-icon"><i class="fas fa-lightbulb"></i></span>
                                <div class="tip-content">${tip}</div>
                            </div>
                        `)
                        .join('');

                    formattedContent += `
                        <div class="diet-tips">
                            <h4><i class="fas fa-lightbulb me-2"></i>Tips for Better Absorption</h4>
                            ${tips}
                        </div>
                    `;
                }
            }

            dietSection.innerHTML = `
                <div class="diet-content">
                    ${formattedContent}
                </div>
            `;
            
            console.log('Formatted diet section:', dietSection.outerHTML);
            
        } catch (error) {
            console.error('Error formatting diet recommendation:', error);
            // Fallback to basic formatting
            dietSection.innerHTML = `
                <div class="diet-header">
                    <h2><i class="fas fa-utensils me-2"></i>Diet Recommendations</h2>
                </div>
                <div class="diet-content">
                    <div class="diet-card">
                        ${recommendation.replace(/\n/g, '<br>')}
                    </div>
                </div>
            `;
        }
        
        return dietSection;
    }

    // Format reported symptoms
    function formatSymptoms(symptoms) {
        const symptomsContainer = document.getElementById('reportedSymptoms');
        symptomsContainer.innerHTML = '';

        if (symptoms.length === 0) {
            symptomsContainer.innerHTML = '<li class="list-group-item text-muted">No symptoms reported</li>';
            return;
        }

        symptoms.forEach((symptom, index) => {
            const li = document.createElement('li');
            li.className = 'list-group-item d-flex align-items-center';
            li.style.animationDelay = `${index * 0.1}s`;
            
            li.innerHTML = `
                <i class="fas fa-check-circle text-success me-2"></i>
                ${symptom}
            `;
            
            symptomsContainer.appendChild(li);
        });
    }

    // Handle form submission
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        // Show loading state
        loadingSpinner.classList.remove('d-none');
        resultsCard.classList.add('d-none');
        errorAlert.classList.add('d-none');

        // Collect form data
        const formData = new FormData(form);
        const data = {};

        // Convert form data to JSON
        for (let [key, value] of formData.entries()) {
            if (key === 'Age') {
                data[key] = parseInt(value);
            } else if (value === 'on') {
                data[key] = true;
            } else if (value === '') {
                data[key] = false;
            } else {
                data[key] = value;
            }
        }

        // Add missing symptom fields as false
        const symptomFields = [
            'Night Blindness', 'Dry Eyes', 'Bleeding Gums', 'Fatigue', 
            'Tingling Sensation', 'Low Sun Exposure', 'Memory Loss',
            'Shortness of Breath', 'Loss of Appetite', 'Fast Heart Rate',
            'Muscle Weakness', 'Weight Loss', 'Reduced Wound Healing Capacity',
            'Bone Pain', 'Depression', 'Weakened Immune System', 'Numbness',
            'Sore Throat', 'Cracked Lips', 'Light Sensitivity', 'Itchy Eyes',
            'Headache', 'Diarrhea', 'Confusion', 'Vision Problems',
            'Poor Balance', 'Easy Bruising', 'Heavy Menstrual Bleeding',
            'Blood in Urine', 'Dark Stool'
        ];

        symptomFields.forEach(field => {
            if (!(field in data)) {
                data[field] = false;
            }
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('API Response:', result);

            // Hide loading spinner
            loadingSpinner.classList.add('d-none');

            // Show and populate results
            resultsCard.classList.remove('d-none');
            requestAnimationFrame(() => {
                resultsCard.classList.add('show');
            });

            // Clear previous results
            const predictionResult = document.getElementById('predictionResult');
            predictionResult.innerHTML = '';

            // Format and display predictions
            formatPredictions(result.top_predictions);
            formatSymptoms(result.reported_symptoms);

            // Add diet recommendations
            if (result.diet_recommendation) {
                console.log('Processing diet recommendation:', result.diet_recommendation);
                
                // If we have structured diet data, use the modern UI
                if (result.diet_data) {
                    try {
                        console.log('Initializing modern diet UI with data:', result.diet_data);
                        initDietPlanUI(result.diet_data, 'dietDashboard');
                    } catch (error) {
                        console.error('Error initializing diet UI:', error);
                        // Fallback to traditional display if modern UI fails
                        const dietSection = formatDietRecommendation(result.diet_recommendation);
                        predictionResult.appendChild(dietSection);
                    }
                } else {
                    // Use traditional markdown display if no structured data
                    const dietSection = formatDietRecommendation(result.diet_recommendation);
                    predictionResult.appendChild(dietSection);
                }
                
                console.log('Diet section added to UI');
            }

            // Scroll to results
            resultsCard.scrollIntoView({ behavior: 'smooth' });

        } catch (error) {
            console.error('Error:', error);
            loadingSpinner.classList.add('d-none');
            showError('An error occurred while processing your request. Please try again.');
        }
    });

    // Initialize the form
    updateProgress();
}); 