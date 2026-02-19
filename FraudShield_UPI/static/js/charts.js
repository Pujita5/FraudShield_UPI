document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById("accuracyChart");
    if (!ctx) return;

    new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Decision Tree", "SVM", "Random Forest", "XGBoost"],
            datasets: [{
                label: "Accuracy (%)",
                data: [83.2, 85.9, 88.1, 99.4],
                borderWidth: 1
            }]
        },
        options: {
            responsive:true,
            scales: {
                y: { beginAtZero: true, max: 100 }
            }
        }
    });
});
const f1Canvas = document.getElementById("f1Chart");
if (f1Canvas) {
    new Chart(f1Canvas, {
        type: "bar",
        data: {
            labels: ["DT", "SVM", "RF", "XGB"],
            datasets: [{
                label: "F1 Score",
                data: [0.80, 0.83, 0.86, 0.99],
            }]
        },
        options: {
            scales: {
                y: { beginAtZero: true, max: 1 }
            }
        }
    });
}
