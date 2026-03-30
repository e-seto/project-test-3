/* ── random pools ── */
const CITIES = [
    { city: "Warrenville", state: "SC", zip: "29851", lat: 33.526,  long: -81.7956,  city_pop: 7524    },
    { city: "Lakewood",    state: "OH", zip: "44107", lat: 41.4847, long: -81.8018,  city_pop: 52244   },
    { city: "Houston",     state: "TX", zip: "77075", lat: 29.6223, long: -95.26,    city_pop: 2906700 },
    { city: "Portland",    state: "OR", zip: "97201", lat: 45.5231, long: -122.6765, city_pop: 652503  },
    { city: "Tucson",      state: "AZ", zip: "85701", lat: 32.2226, long: -110.9747, city_pop: 542629  },
    { city: "Durham",      state: "NC", zip: "27701", lat: 35.994,  long: -78.8986,  city_pop: 278993  },
    { city: "Richmond",    state: "VA", zip: "23220", lat: 37.5407, long: -77.436,   city_pop: 226610  },
    { city: "Boise",       state: "ID", zip: "83702", lat: 43.615,  long: -116.2023, city_pop: 235684  },
    { city: "Nevada",      state: "MO", zip: "64772", lat: 37.7749, long: -94.3571,  city_pop: 13596   },
];

const JOBS = ["Art gallery manager","Accountant, chartered","Airline pilot","Architect","Barrister","Civil engineer, consulting","Data scientist","Dentist","Economist","Firefighter","Graphic designer","Hotel manager","IT consultant","Journalist, newspaper","Lawyer","Marketing officer","Mechanical engineer","Nurse, adult","Occupational therapist","Pharmacist, hospital","Police officer","Psychologist, clinical","Research scientist (medical)","Social worker","Software engineer","Surgeon","Teacher, secondary school","Theatre manager","Veterinary surgeon","Web designer"];

/* ── current fixed defaults ── */
let currentDefaults = {
    city:      "Warrenville",
    state:     "SC",
    zip:       "29851",
    lat:       33.526,
    long:      -81.7956,
    city_pop:  7524,
    job:       "Art gallery manager",
    haversine_km: 36.46,
};

/* ── randomize fixed defaults ── */
function randomizeDefaults() {
    const c = CITIES[Math.floor(Math.random() * CITIES.length)];
    const job = JOBS[Math.floor(Math.random() * JOBS.length)];

    // random merchant coords near city for haversine
    const merch_lat  = c.lat  + (Math.random() - 0.5) * 1.5;
    const merch_long = c.long + (Math.random() - 0.5) * 1.5;

    // haversine calculation
    const dlat = (merch_lat - c.lat) * Math.PI / 180;
    const dlon = (merch_long - c.long) * Math.PI / 180;
    const a = Math.sin(dlat/2)**2 + Math.cos(c.lat*Math.PI/180) * Math.cos(merch_lat*Math.PI/180) * Math.sin(dlon/2)**2;
    const haversine_km = parseFloat((6371 * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a))).toFixed(2));

    currentDefaults = {
        city: c.city, state: c.state, zip: c.zip,
        lat: c.lat, long: c.long, city_pop: c.city_pop,
        job, haversine_km,
    };

    updateDataDisplay();
}

/* ── field display labels ── */
const FIELD_LABELS = [
    ["city",         "City:"],
    ["state",        "State:"],
    ["zip",          "ZIP:"],
    ["city_pop",     "City Population:"],
    ["job",          "Job:"],
    ["haversine_km", "Distance to Merchant (km):"],
];

/* ── update the data summary panel ── */
function updateDataDisplay() {
    const box = document.getElementById("data-display");
    box.innerHTML = FIELD_LABELS.map(([key, label]) =>
        `<div class="data-row">
            <span class="data-label">${label}</span>
            <span class="data-value">${currentDefaults[key]}</span>
        </div>`
    ).join("");
}

/* ── collect user inputs + merge with fixed defaults ── */
function buildPayload() {
    const amt           = parseFloat(document.getElementById("amt").value);
    const cust_amt_mean = parseFloat(document.getElementById("cust_amt_mean").value);
    return {
        ...currentDefaults,
        gender:        document.getElementById("gender").value,
        category:      document.getElementById("category").value,
        day_of_week:   document.getElementById("day_of_week").value,
        hour:          parseInt(document.getElementById("hour").value),
        month:         parseInt(document.getElementById("month").value),
        age:           parseInt(document.getElementById("age").value),
        amt:           amt,
        cust_amt_mean: cust_amt_mean,
        amt_dev_from_mean: amt - cust_amt_mean,
    };
}

/* ── predict ── */
function predict() {
    const resultBox = document.getElementById("prediction-result");
    resultBox.innerHTML = "Running model...";

    fetch("/predictive", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPayload())
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            resultBox.innerHTML = `⚠️ ${data.error}`;
            return;
        }

        const prob = (data.fraud_probability * 100).toFixed(2);

        let decisionText  = "";
        let decisionClass = "";

        if (data.decision === "legit") {
            decisionText  = "Legit Transaction";
            decisionClass = "safe";
        } else if (data.decision === "manual_review") {
            decisionText  = "Needs Manual Review";
            decisionClass = "review";
        } else {
            decisionText  = "Block Transaction";
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

/* ── initialize ── */
document.addEventListener("DOMContentLoaded", () => {
    updateDataDisplay();

    document.getElementById("refresh-btn").addEventListener("click", randomizeDefaults);
    document.getElementById("predict-btn").addEventListener("click", predict);
});
