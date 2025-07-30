export function makePolarChart (logitArray, ctx, isLive) {
    if (window.polarChartInstance) {
        window.polarChartInstance.destroy();
    }
    let chartTitle = '';
    if (isLive) {
      chartTitle = "Live Classification";
    } else {
      chartTitle = "Overall Classification";
    }
    const polarLabelPlugin = {
        id: 'polarLabelPlugin',
        afterDatasetDraw(chart) {
            const { ctx, data, chartArea, scales } = chart;
            const meta = chart.getDatasetMeta(0);
        
            const centerX = scales.r.xCenter;
            const centerY = scales.r.yCenter;
        
            const maxRadius = scales.r.drawingArea;
            const labelOffset = 60;
            const radius = maxRadius + labelOffset;
        
            meta.data.forEach((arc, index) => {
                const label = data.labels[index];
                if (!label) return;
        
                const angle = (arc.startAngle + arc.endAngle) / 2;
                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;
        
                ctx.save();
                ctx.font = '12px sans-serif';
                ctx.fillStyle = 'black';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, x, y);
                ctx.restore();
            });
        }
    };
      
    window.polarChartInstance = new Chart(ctx, {
        type: 'polarArea',
        data: {
            labels : [
                'bad-attack-clarity',
                'bad-dynamic-stability',
                'bad-pitch-stability',
                'bad-timbre-richness',
                'bad-timbre-stability',
                'good-sound'
            ],
            datasets: [{
                label: 'Confidence Scores',
                data: Array.from(logitArray),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(75, 192, 192, 0.2)'
                  ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)',
                    'rgba(75, 192, 192, 1)',
                  ],
                pointBackgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(75, 192, 192, 0.2)'
                  ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            layout : {
                padding: {
                    top: 20,
                    bottom: 20,
                    left: 20,
                    right: 20
                }
            },
            plugins: {
                title: {
                  display: true,
                  text: chartTitle,
                  font: {
                    size: 20,
                    weight: 'bold'
                  }
                },
                legend: {
                  display: false,
                  position: 'bottom'
                },
                tooltip: {
                  callbacks: {
                    label: function(context) {
                      return `${context.label}: ${Number(context.parsed).toFixed(3)}`;
                    }
                  }
                }
            },
            scales: {
                r: {
                  beginAtZero: true,
                  min: 0,
                  max: 1,
                  ticks: {
                    stepSize: 0.2
                  },
                  pointLabels: {
                    display: false
                  }
                }
            },
            animation: {
                duration: 0 
            },
        },
        // plugins: [polarLabelPlugin]
    });
}

export function makeLineChart(probabilities, timestamps, ctx) {
  const numClasses = probabilities[0].length;
  const labels = timestamps;

  const datasets = Array.from({length: numClasses}, (_, classIndex) => ({
    label: classLabels[classIndex],
    data: probabilities.map(row => row[classIndex]),
    borderColor : labelColors[classIndex % labelColors.length],
    backgroundColor: labelColors[classIndex % labelColors.length]
  }));

  if (window.lineChartInstance) {
    window.lineChartInstance.destroy();
  }

  window.lineChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: datasets
    },
    options: {
      responsive : true,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Time (seconds)'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Confidence'
          }
        }
      },
      plugins : {
        legend: {
          display: false,
          position: 'bottom'
        },
        title: {
          display : true,
          text: 'Prediction Confidence Over Time'
        }
      }
    }
  });
}

export const labelColors = [
  'rgba(255, 99, 132, 1)',
  'rgba(54, 162, 235, 1)',
  'rgba(255, 206, 86, 1)',
  'rgba(153, 102, 255, 1)',
  'rgba(255, 159, 64, 1)',
  'rgba(75, 192, 192, 1)'
]

export const classLabels = [
  'bad-attack-clarity',
  'bad-dynamic-stability',
  'bad-pitch-stability',
  'bad-timbre-richness',
  'bad-timbre-stability',
  'good-sound'
]

export function renderLegend() {
  const legendContainer = document.getElementById('chartLegend');
  const wrapper = legendContainer.closest('.legend-container');
  if (wrapper) wrapper.style.display = 'block';

  legendContainer.innerHTML = ''; // Clear existing content

  classLabels.forEach((label, index) => {
    const legendItem = document.createElement('div');
    legendItem.style.display = 'flex';
    legendItem.style.alignItems = 'center';
    legendItem.style.gap = '6px';

    const colorBox = document.createElement('span');
    colorBox.style.display = 'inline-block';
    colorBox.style.width = '12px';
    colorBox.style.height = '12px';
    colorBox.style.backgroundColor = labelColors[index];
    colorBox.style.borderRadius = '2px';

    const text = document.createElement('span');
    text.textContent = label;

    legendItem.appendChild(colorBox);
    legendItem.appendChild(text);
    legendContainer.appendChild(legendItem);
  });
}

export function getAverageProbabilities(predictions) {
  if (!predictions || predictions.length === 0) return [];

  const numClasses = predictions[0].length;
  const avgProbs = new Array(numClasses).fill(0);

  predictions.forEach(probs => {
    probs.forEach((p, i) => {
      avgProbs[i] += p;
    });
  });

  return avgProbs.map(p => p / predictions.length);
}