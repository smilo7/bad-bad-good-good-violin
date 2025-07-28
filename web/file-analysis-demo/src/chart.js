export function makePolarChart (logitArray, ctx) {
    if (window.polarChartInstance) {
        window.polarChartInstance.destroy();
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
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
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
                    left: 120,
                    right: 120
                }
            },
            plugins: {
                legend: {
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
                  }
                }
            },
            animation: {
                animateRotate: true
            },
        },
        plugins: [polarLabelPlugin]
    });
     
}

export function makeLineChart(probabilities, ctx) {
  const numClasses = probabilities[0].length;

  const labels = probabilities[0].map((_, i) => i);
  
  const datasets = Array.from({length: numClasses}, (_, classIndex) => ({
    label: classLabels[classIndex],
    data: probabilities.map(row => row[classIndex]),
    borderColor : labelColors[classIndex % labelColors.length],
    backgroundColor: labelColors[classIndex % labelColors.length]
  }))


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
      plugins : {
        title: {
          display : true,
          text : 'Line Chart'
        }
      }
    }
  })

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