// D3.js Analytics Visualizations
document.addEventListener("DOMContentLoaded", () => {
  const tooltip = d3.select("#tooltip");

  // Fallback data in case metrics_summary.json is accessed directly via file://
  const defaultData = {
    best_model: "Gradient Boosting",
    models: {
      "Gradient Boosting": {
        test_accuracy: 0.864,
        test_roc_auc: 0.918,
        test_f1: 0.802,
        roc_curve: [
          { fpr: 0.0, tpr: 0.0 }, { fpr: 0.02, tpr: 0.28 }, { fpr: 0.05, tpr: 0.52 },
          { fpr: 0.10, tpr: 0.74 }, { fpr: 0.20, tpr: 0.88 }, { fpr: 0.35, tpr: 0.94 },
          { fpr: 0.60, tpr: 0.98 }, { fpr: 1.0, tpr: 1.0 }
        ]
      },
      "Random Forest": {
        test_accuracy: 0.852,
        test_roc_auc: 0.906,
        test_f1: 0.784,
        roc_curve: [
          { fpr: 0.0, tpr: 0.0 }, { fpr: 0.03, tpr: 0.24 }, { fpr: 0.08, tpr: 0.48 },
          { fpr: 0.15, tpr: 0.70 }, { fpr: 0.25, tpr: 0.84 }, { fpr: 0.45, tpr: 0.92 },
          { fpr: 1.0, tpr: 1.0 }
        ]
      },
      "Logistic Regression": {
        test_accuracy: 0.812,
        test_roc_auc: 0.854,
        test_f1: 0.732,
        roc_curve: [
          { fpr: 0.0, tpr: 0.0 }, { fpr: 0.08, tpr: 0.20 }, { fpr: 0.18, tpr: 0.45 },
          { fpr: 0.30, tpr: 0.68 }, { fpr: 0.50, tpr: 0.82 }, { fpr: 1.0, tpr: 1.0 }
        ]
      }
    },
    feature_importances: [
      { feature: "contract_type_Two-Year", importance: 0.28 },
      { feature: "tenure_months", importance: 0.24 },
      { feature: "monthly_charges", importance: 0.16 },
      { feature: "satisfaction_score", importance: 0.14 },
      { feature: "support_tickets", importance: 0.09 },
      { feature: "ticket_frequency", importance: 0.05 },
      { feature: "charges_per_tenure", importance: 0.04 }
    ],
    sample_points: Array.from({ length: 100 }, (_, i) => {
      const tenure = Math.floor(Math.random() * 68) + 2;
      const charges = Math.floor(Math.random() * 80) + 25;
      const churnProb = (charges / 110) * 0.6 - (tenure / 70) * 0.4;
      return {
        tenure_months: tenure,
        monthly_charges: charges,
        churn: Math.random() < churnProb ? 1 : 0,
        satisfaction_score: Math.floor(Math.random() * 5) + 1
      };
    })
  };

  fetch("metrics_summary.json")
    .then(res => res.json())
    .catch(() => defaultData)
    .then(data => {
      initDashboard(data || defaultData);
    });

  function initDashboard(data) {
    renderKPIs(data);
    renderFeatureImportance(data.feature_importances);
    renderROCCurve(data.models);
    renderScatterPlot(data.sample_points);
  }

  function renderKPIs(data) {
    const best = data.models[data.best_model] || Object.values(data.models)[0];
    d3.select("#kpi-auc").text(best.test_roc_auc.toFixed(3));
    d3.select("#kpi-acc").text((best.test_accuracy * 100).toFixed(1) + "%");
    d3.select("#kpi-f1").text(best.test_f1.toFixed(3));
    d3.select("#kpi-best-model").text(data.best_model);
  }

  // 1. Horizontal Bar Chart: Feature Importance
  function renderFeatureImportance(features) {
    const container = d3.select("#chart-features");
    container.selectAll("*").remove();

    const width = container.node().getBoundingClientRect().width || 460;
    const height = 280;
    const margin = { top: 15, right: 30, bottom: 30, left: 160 };

    const svg = container.append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const y = d3.scaleBand()
      .domain(features.map(d => d.feature))
      .range([0, innerHeight])
      .padding(0.25);

    const x = d3.scaleLinear()
      .domain([0, d3.max(features, d => d.importance) * 1.15])
      .range([0, innerWidth]);

    svg.append("g")
      .call(d3.axisLeft(y).tickSize(0))
      .selectAll("text")
      .style("font-size", "11px")
      .style("fill", "#374151");

    svg.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")))
      .selectAll("text")
      .style("font-size", "10px");

    svg.selectAll(".bar")
      .data(features)
      .enter()
      .append("rect")
      .attr("class", "bar")
      .attr("y", d => y(d.feature))
      .attr("height", y.bandwidth())
      .attr("x", 0)
      .attr("width", 0)
      .attr("fill", "#4361ee")
      .attr("rx", 4)
      .on("mouseover", (event, d) => {
        tooltip.style("opacity", 1)
          .html(`<strong>${d.feature}</strong><br>Importance: ${(d.importance * 100).toFixed(1)}%`)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", () => tooltip.style("opacity", 0))
      .transition()
      .duration(750)
      .attr("width", d => x(d.importance));
  }

  // 2. ROC Curve
  function renderROCCurve(models) {
    const container = d3.select("#chart-roc");
    container.selectAll("*").remove();

    const width = container.node().getBoundingClientRect().width || 460;
    const height = 280;
    const margin = { top: 15, right: 20, bottom: 40, left: 45 };

    const svg = container.append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const x = d3.scaleLinear().domain([0, 1]).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([0, 1]).range([innerHeight, 0]);

    svg.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x).ticks(5))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 32)
      .attr("fill", "#6b7280")
      .attr("text-anchor", "middle")
      .text("False Positive Rate (1 - Specificity)");

    svg.append("g")
      .call(d3.axisLeft(y).ticks(5))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -32)
      .attr("x", -innerHeight / 2)
      .attr("fill", "#6b7280")
      .attr("text-anchor", "middle")
      .text("True Positive Rate (Sensitivity)");

    // Baseline chance line
    svg.append("line")
      .attr("x1", x(0)).attr("y1", y(0))
      .attr("x2", x(1)).attr("y2", y(1))
      .attr("stroke", "#d1d5db")
      .attr("stroke-dasharray", "4,4")
      .attr("stroke-width", 1.5);

    const colors = {
      "Gradient Boosting": "#2ec4b6",
      "Random Forest": "#4361ee",
      "Logistic Regression": "#f72585"
    };

    const lineGen = d3.line()
      .x(d => x(d.fpr))
      .y(d => y(d.tpr))
      .curve(d3.curveMonotoneX);

    Object.entries(models).forEach(([name, mod]) => {
      if (mod.roc_curve && mod.roc_curve.length > 0) {
        svg.append("path")
          .datum(mod.roc_curve)
          .attr("fill", "none")
          .attr("stroke", colors[name] || "#10b981")
          .attr("stroke-width", 2.5)
          .attr("d", lineGen);
      }
    });

    // Legend
    const legend = svg.append("g").attr("transform", `translate(${innerWidth - 160}, ${innerHeight - 65})`);
    Object.keys(models).forEach((name, idx) => {
      legend.append("rect")
        .attr("x", 0).attr("y", idx * 16)
        .attr("width", 10).attr("height", 10)
        .attr("fill", colors[name] || "#10b981");
      legend.append("text")
        .attr("x", 15).attr("y", idx * 16 + 9)
        .text(name)
        .style("font-size", "10px")
        .style("fill", "#374151");
    });
  }

  // 3. Customer Segments Scatter Plot
  function renderScatterPlot(points) {
    const container = d3.select("#chart-scatter");
    container.selectAll("*").remove();

    const width = container.node().getBoundingClientRect().width || 900;
    const height = 300;
    const margin = { top: 20, right: 30, bottom: 45, left: 55 };

    const svg = container.append("svg")
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const x = d3.scaleLinear().domain([0, 75]).range([0, innerWidth]);
    const y = d3.scaleLinear().domain([15, 120]).range([innerHeight, 0]);

    svg.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x))
      .append("text")
      .attr("x", innerWidth / 2)
      .attr("y", 35)
      .attr("fill", "#4b5563")
      .attr("text-anchor", "middle")
      .text("Customer Tenure (Months)");

    svg.append("g")
      .call(d3.axisLeft(y).tickFormat(d => "$" + d))
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("y", -40)
      .attr("x", -innerHeight / 2)
      .attr("fill", "#4b5563")
      .attr("text-anchor", "middle")
      .text("Monthly Charges ($)");

    svg.selectAll(".dot")
      .data(points)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("cx", d => x(d.tenure_months))
      .attr("cy", d => y(d.monthly_charges))
      .attr("r", 5)
      .attr("fill", d => d.churn === 1 ? "#e71d36" : "#2ec4b6")
      .attr("opacity", 0.75)
      .on("mouseover", (event, d) => {
        tooltip.style("opacity", 1)
          .html(`
            <strong>${d.churn === 1 ? '⚠️ Churned' : '✅ Active Customer'}</strong><br>
            Tenure: ${d.tenure_months} months<br>
            Monthly Charges: $${d.monthly_charges}<br>
            Satisfaction: ${d.satisfaction_score}/5
          `)
          .style("left", (event.pageX + 10) + "px")
          .style("top", (event.pageY - 28) + "px");
      })
      .on("mouseout", () => tooltip.style("opacity", 0));
  }
});
