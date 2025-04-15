/**
 * Modern Diet Plan UI Handler
 * This script creates an interactive UI for displaying diet plans using the structured JSON data from the API
 */

class DietPlanUI {
    constructor(dietData, container) {
        this.dietData = dietData;
        this.container = container;
        this.mealIcons = {
            'Breakfast': 'fa-coffee',
            'Mid-morning Snack': 'fa-apple-alt',
            'Lunch': 'fa-utensils',
            'Evening Snack': 'fa-cookie',
            'Dinner': 'fa-drumstick-bite'
        };
    }

    /**
     * Render the complete diet dashboard
     */
    render() {
        if (!this.dietData) {
            console.warn('No diet data available');
            return this.renderFallback();
        }

        try {
            const dashboard = document.createElement('div');
            dashboard.className = 'diet-dashboard';

            // Add header
            dashboard.appendChild(this.createHeader());

            // Create tabs for different sections
            const tabsNav = document.createElement('ul');
            tabsNav.className = 'nav nav-tabs diet-tabs';
            tabsNav.setAttribute('role', 'tablist');
            tabsNav.id = 'dietTabs';

            const tabContent = document.createElement('div');
            tabContent.className = 'tab-content diet-content';
            tabContent.id = 'dietTabContent';

            // Add meal plans tab
            this.addTab(tabsNav, tabContent, 'meals', 'Daily Meals', 'fa-utensils', true);
            
            // Add nutrients tab
            this.addTab(tabsNav, tabContent, 'nutrients', 'Key Nutrients', 'fa-seedling');
            
            // Add foods to avoid tab
            this.addTab(tabsNav, tabContent, 'avoid', 'Foods to Avoid', 'fa-ban');
            
            // Add supplements tab
            this.addTab(tabsNav, tabContent, 'supplements', 'Supplements', 'fa-pills');
            
            // Add tips tab
            this.addTab(tabsNav, tabContent, 'tips', 'Absorption Tips', 'fa-lightbulb');

            dashboard.appendChild(tabsNav);
            dashboard.appendChild(tabContent);

            // Empty container and append dashboard
            this.container.innerHTML = '';
            this.container.appendChild(dashboard);
            
            // Initialize the tab content
            this.renderMealPlans(document.getElementById('meals-tab-pane'));
            this.renderNutrients(document.getElementById('nutrients-tab-pane'));
            this.renderFoodsToAvoid(document.getElementById('avoid-tab-pane'));
            this.renderSupplements(document.getElementById('supplements-tab-pane'));
            this.renderTips(document.getElementById('tips-tab-pane'));

            // Initialize Bootstrap tabs
            const triggerTabList = Array.from(tabsNav.querySelectorAll('button'));
            triggerTabList.forEach(triggerEl => {
                const tabTrigger = new bootstrap.Tab(triggerEl);
                triggerEl.addEventListener('click', event => {
                    event.preventDefault();
                    tabTrigger.show();
                });
            });
        } catch (error) {
            console.error('Error rendering diet dashboard:', error);
            this.renderFallback();
        }
    }

    /**
     * Create the diet plan header
     */
    createHeader() {
        const header = document.createElement('div');
        header.className = 'diet-header';
        
        const title = document.createElement('h2');
        title.innerHTML = `<i class="fas fa-utensils me-2"></i>${this.dietData.title || 'Personalized Diet Plan'}`;
        
        const intro = document.createElement('p');
        intro.textContent = this.dietData.introduction || '';
        
        header.appendChild(title);
        header.appendChild(intro);
        
        return header;
    }

    /**
     * Add a tab to the tabs navigation and content
     */
    addTab(tabsNav, tabContent, id, label, icon, isActive = false) {
        // Create tab navigation item
        const navItem = document.createElement('li');
        navItem.className = 'nav-item';
        navItem.setAttribute('role', 'presentation');
        
        const navButton = document.createElement('button');
        navButton.className = `nav-link ${isActive ? 'active' : ''}`;
        navButton.id = `${id}-tab`;
        navButton.setAttribute('data-bs-toggle', 'tab');
        navButton.setAttribute('data-bs-target', `#${id}-tab-pane`);
        navButton.setAttribute('type', 'button');
        navButton.setAttribute('role', 'tab');
        navButton.setAttribute('aria-controls', `${id}-tab-pane`);
        navButton.setAttribute('aria-selected', isActive ? 'true' : 'false');
        navButton.innerHTML = `<i class="fas ${icon} me-1"></i> ${label}`;
        
        navItem.appendChild(navButton);
        tabsNav.appendChild(navItem);
        
        // Create tab content
        const tabPane = document.createElement('div');
        tabPane.className = `tab-pane fade ${isActive ? 'show active' : ''}`;
        tabPane.id = `${id}-tab-pane`;
        tabPane.setAttribute('role', 'tabpanel');
        tabPane.setAttribute('aria-labelledby', `${id}-tab`);
        tabPane.setAttribute('tabindex', '0');
        
        tabContent.appendChild(tabPane);
    }

    /**
     * Render daily meal plans
     */
    renderMealPlans(container) {
        if (!this.dietData.daily_meals || this.dietData.daily_meals.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No meal plan information available.</div>';
            return;
        }

        const mealsContainer = document.createElement('div');
        mealsContainer.className = 'meal-plans-container';
        
        this.dietData.daily_meals.forEach(meal => {
            const card = document.createElement('div');
            card.className = 'meal-card';
            
            // Meal header
            const header = document.createElement('div');
            header.className = 'meal-header';
            
            const icon = document.createElement('div');
            icon.className = 'meal-icon';
            icon.innerHTML = `<i class="fas ${this.mealIcons[meal.name] || 'fa-utensils'}"></i>`;
            
            const title = document.createElement('h5');
            title.className = 'meal-title';
            title.textContent = meal.name;
            
            header.appendChild(icon);
            header.appendChild(title);
            
            // Meal body
            const body = document.createElement('div');
            body.className = 'meal-body';
            
            const description = document.createElement('p');
            description.className = 'meal-description';
            description.textContent = meal.description;
            
            const suggestionsList = document.createElement('ul');
            suggestionsList.className = 'meal-suggestions';
            
            meal.food_suggestions.forEach(suggestion => {
                const li = document.createElement('li');
                li.textContent = suggestion;
                suggestionsList.appendChild(li);
            });
            
            body.appendChild(description);
            body.appendChild(suggestionsList);
            
            // Assemble card
            card.appendChild(header);
            card.appendChild(body);
            mealsContainer.appendChild(card);
        });
        
        container.appendChild(mealsContainer);
    }

    /**
     * Render key nutrients section
     */
    renderNutrients(container) {
        if (!this.dietData.key_nutrients || this.dietData.key_nutrients.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No nutrient information available.</div>';
            return;
        }

        const nutrientsContainer = document.createElement('div');
        nutrientsContainer.className = 'nutrients-container';
        
        this.dietData.key_nutrients.forEach(nutrient => {
            const card = document.createElement('div');
            card.className = 'nutrient-card';
            
            const name = document.createElement('div');
            name.className = 'nutrient-name';
            name.textContent = nutrient.name;
            
            const benefits = document.createElement('div');
            benefits.className = 'nutrient-benefits';
            benefits.textContent = nutrient.benefits;
            
            const sources = document.createElement('div');
            sources.className = 'nutrient-sources';
            
            const sourcesTitle = document.createElement('h6');
            sourcesTitle.textContent = 'Food Sources';
            
            const sourceTags = document.createElement('div');
            sourceTags.className = 'source-tags';
            
            nutrient.food_sources.forEach(source => {
                const tag = document.createElement('span');
                tag.className = 'source-tag';
                tag.textContent = source;
                sourceTags.appendChild(tag);
            });
            
            sources.appendChild(sourcesTitle);
            sources.appendChild(sourceTags);
            
            card.appendChild(name);
            card.appendChild(benefits);
            card.appendChild(sources);
            
            nutrientsContainer.appendChild(card);
        });
        
        container.appendChild(nutrientsContainer);
    }

    /**
     * Render foods to avoid section
     */
    renderFoodsToAvoid(container) {
        if (!this.dietData.foods_to_avoid || this.dietData.foods_to_avoid.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No foods to avoid information available.</div>';
            return;
        }

        const avoidContainer = document.createElement('div');
        avoidContainer.className = 'avoid-container';
        
        this.dietData.foods_to_avoid.forEach(food => {
            const card = document.createElement('div');
            card.className = 'avoid-card';
            
            const icon = document.createElement('div');
            icon.className = 'avoid-icon';
            icon.innerHTML = '<i class="fas fa-ban"></i>';
            
            const content = document.createElement('div');
            content.className = 'avoid-content';
            
            const name = document.createElement('div');
            name.className = 'avoid-name';
            name.textContent = food.name;
            
            const reason = document.createElement('div');
            reason.className = 'avoid-reason';
            reason.textContent = food.reason;
            
            content.appendChild(name);
            content.appendChild(reason);
            
            card.appendChild(icon);
            card.appendChild(content);
            
            avoidContainer.appendChild(card);
        });
        
        container.appendChild(avoidContainer);
    }

    /**
     * Render supplements section
     */
    renderSupplements(container) {
        if (!this.dietData.supplements || this.dietData.supplements.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No supplement information available.</div>';
            return;
        }

        const supplementsContainer = document.createElement('div');
        supplementsContainer.className = 'supplements-container';
        
        this.dietData.supplements.forEach(supplement => {
            const card = document.createElement('div');
            card.className = 'supplement-card';
            
            const header = document.createElement('div');
            header.className = 'supplement-header';
            header.textContent = supplement.name;
            
            const body = document.createElement('div');
            body.className = 'supplement-body';
            
            const dosage = document.createElement('div');
            dosage.className = 'supplement-dosage';
            dosage.innerHTML = `<i class="fas fa-prescription-bottle me-2"></i>${supplement.dosage}`;
            
            const notes = document.createElement('div');
            notes.className = 'supplement-notes';
            notes.textContent = supplement.notes;
            
            body.appendChild(dosage);
            body.appendChild(notes);
            
            card.appendChild(header);
            card.appendChild(body);
            
            supplementsContainer.appendChild(card);
        });
        
        container.appendChild(supplementsContainer);
    }

    /**
     * Render absorption tips section
     */
    renderTips(container) {
        if (!this.dietData.absorption_tips || this.dietData.absorption_tips.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No absorption tips available.</div>';
            return;
        }

        const tipsContainer = document.createElement('div');
        tipsContainer.className = 'tips-container';
        
        const header = document.createElement('div');
        header.className = 'tips-header';
        header.innerHTML = '<i class="fas fa-lightbulb"></i><h4>Tips for Better Absorption</h4>';
        
        const tipsList = document.createElement('div');
        tipsList.className = 'tips-list';
        
        this.dietData.absorption_tips.forEach(tip => {
            const card = document.createElement('div');
            card.className = 'tip-card';
            
            const icon = document.createElement('div');
            icon.className = 'tip-icon';
            icon.innerHTML = '<i class="fas fa-check"></i>';
            
            const content = document.createElement('div');
            content.className = 'tip-content';
            content.textContent = tip;
            
            card.appendChild(icon);
            card.appendChild(content);
            
            tipsList.appendChild(card);
        });
        
        tipsContainer.appendChild(header);
        tipsContainer.appendChild(tipsList);
        
        container.appendChild(tipsContainer);
    }

    /**
     * Render a fallback interface for when structured data isn't available
     */
    renderFallback() {
        this.container.innerHTML = `
            <div class="alert alert-warning">
                <i class="fas fa-exclamation-triangle me-2"></i>
                Unable to load advanced diet interface. Please try again later.
            </div>
        `;
    }
}

// Function to initialize diet plan UI
function initDietPlanUI(dietData, containerId) {
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`Container with ID "${containerId}" not found`);
        return;
    }
    
    const ui = new DietPlanUI(dietData, container);
    ui.render();
} 