/* transaction data */
const transactions = {
    1: { // 0
        "gender": "M",
        "city": "Warrenville",
        "state": "SC",
        "zip": "29851",
        "lat": 33.526,
        "long": -81.7956,
        "city_pop": 7524,
        "job": "Art gallery manager",
        "category": "personal_care",
        "amt": 5.23,
        "merchant": "Homenick LLC",
        "merch_lat": 33.218923,
        "merch_long": -81.657941,
        "age": 53,
        "hour": 14,
        "day_of_week": "Sunday",
        "month": 11,
        "haversine_km": 36.45976,
        "cust_amt_mean": 65.014847,
        "amt_dev_from_mean": -59.784847
    },
    2: { // 1
        "gender": "F",
        "city": "Lakewood",
        "state": "OH",
        "zip": "44107",
        "lat": 41.4847,
        "long": -81.8018,
        "city_pop": 52244,
        "job": "Theatre manager",
        "category": "shopping_net",
        "amt": 913.11,
        "merchant": "Zboncak, Rowe and Murazik",
        "merch_lat": 42.158248,
        "merch_long": -81.19596,
        "age": 55,
        "hour": 19,
        "day_of_week": "Wednesday",
        "month": 3,
        "haversine_km": 90.163887,
        "cust_amt_mean": 980.868571,
        "amt_dev_from_mean": -67.758571
    },
    3: { // 1
        "gender": "M",
        "city": "Houston",
        "state": "TX",
        "zip": "77075",
        "lat": 29.6223,
        "long": -95.26,
        "city_pop": 2906700,
        "job": "Operations geologist",
        "category": "grocery_pos",
        "amt": 14.37,
        "merchant": "Bauch-Raynor",
        "merch_lat": 29.979782,
        "merch_long": -95.445867,
        "age": 41,
        "hour": 1,
        "day_of_week": "Sunday",
        "month": 7,
        "haversine_km": 43.608659,
        "cust_amt_mean": 61.886955,
        "amt_dev_from_mean": -47.516955
    },
    4: { // 0
        "gender": "F",
        "city": "Nevada",
        "state": "M0",
        "zip": "64772",
        "lat": 37.7749,
        "long": -94.3571,
        "city_pop": 13596,
        "job": "Commissioning editor",
        "category": "entertainment",
        "amt": 331.65,
        "merchant": "Stark-Batz",
        "merch_lat": 36.928269,
        "merch_long": -95.327865,
        "age": 21,
        "hour": 19,
        "day_of_week": "Tuesday",
        "month": 2,
        "haversine_km": 132.869052,
        "cust_amt_mean": 88.789291,
        "amt_dev_from_mean": 242.860709
    }
};

let currentTransaction = 1;

/* display JSON */
function displayJSON() {
    document.getElementById("jsonDisplay").textContent =
        JSON.stringify(transactions[currentTransaction], null, 2);
}

/* switching tabs */
function selectTab(index) {
    currentTransaction = index;

    document.querySelectorAll(".tab").forEach((tab, i) => {
        tab.classList.toggle("active-tab", i === index - 1);
    });

    displayJSON();
}

/* predict */
function predict() {
    const resultBox = document.getElementById("prediction-result");
    resultBox.innerHTML = "Running model...";

    const transactionData = transactions[currentTransaction];

    fetch("/predictive", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(transactionData)
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            resultBox.innerHTML = `⚠️ ${data.error}`;
            return;
        }

        const prob = (data.fraud_probability * 100).toFixed(2);

        let decisionText = "";
        let decisionClass = "";

        if (data.decision === "legit") {
            decisionText = "Legit Transaction";
            decisionClass = "safe";
        } else if (data.decision === "manual_review") {
            decisionText = "Needs Manual Review";
            decisionClass = "review";
        } else {
            decisionText = "Block Transaction";
            decisionClass = "fraud";
        }

        resultBox.innerHTML = `
            <p><strong>Fraud Probability:</strong> ${prob}%</p>
            <p class="${decisionClass}">${decisionText}</p>
        `;
    })
    .catch(err => {
        console.error(err);
        resultBox.innerHTML = "Server error";
    });
}

/* initialize */
document.addEventListener("DOMContentLoaded", () => {
    displayJSON();

    document
        .getElementById("predict-btn")
        .addEventListener("click", predict);
});