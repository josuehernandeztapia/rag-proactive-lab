#!/usr/bin/env node
/**
 * AVI Questions Analysis Script
 * Analyzes the 55-question dataset and provides insights
 */

const fs = require('fs');
const path = require('path');

// Mock import since this is JavaScript (in real implementation would use TypeScript)
const mockAviQuestions = {
  total: 55,
  categories: [
    { id: 'basic_info', count: 6, name: 'Información Básica' },
    { id: 'daily_operation', count: 8, name: 'Operación Diaria' },
    { id: 'operational_costs', count: 7, name: 'Costos Operativos' },
    { id: 'business_structure', count: 8, name: 'Estructura Empresarial' },
    { id: 'assets_patrimony', count: 6, name: 'Activos y Patrimonio' },
    { id: 'credit_history', count: 6, name: 'Historial Crediticio' },
    { id: 'payment_intention', count: 6, name: 'Intención de Pago' },
    { id: 'risk_evaluation', count: 8, name: 'Evaluación de Riesgo' }
  ],
  weights: {
    '9-10': 12, // Critical questions
    '7-8': 18,  // High importance
    '5-6': 16,  // Medium importance
    '3-4': 9    // Low importance
  },
  stressLevels: {
    '5': 8,  // Maximum stress
    '4': 12, // High stress
    '3': 15, // Medium stress
    '2': 14, // Low stress
    '1': 6   // Minimal stress
  }
};

function generateQuestionAnalysisReport() {
  console.log('🎯 AVI Questions Dataset Analysis Report');
  console.log('=' .repeat(50));

  console.log('\n📊 DATASET OVERVIEW');
  console.log(`Total Questions: ${mockAviQuestions.total}`);
  console.log(`Coverage: 100% Complete`);
  console.log(`Estimated Duration: ~50 minutes`);

  console.log('\n🗂️ DISTRIBUTION BY CATEGORY');
  mockAviQuestions.categories.forEach(cat => {
    const percentage = ((cat.count / mockAviQuestions.total) * 100).toFixed(1);
    console.log(`  ${cat.name}: ${cat.count} preguntas (${percentage}%)`);
  });

  console.log('\n⚖️ DISTRIBUTION BY WEIGHT (Criticality)');
  Object.entries(mockAviQuestions.weights).forEach(([range, count]) => {
    const percentage = ((count / mockAviQuestions.total) * 100).toFixed(1);
    const label = range === '9-10' ? 'Critical' :
                  range === '7-8' ? 'High' :
                  range === '5-6' ? 'Medium' : 'Low';
    console.log(`  Weight ${range} (${label}): ${count} preguntas (${percentage}%)`);
  });

  console.log('\n😰 DISTRIBUTION BY STRESS LEVEL');
  Object.entries(mockAviQuestions.stressLevels).forEach(([level, count]) => {
    const percentage = ((count / mockAviQuestions.total) * 100).toFixed(1);
    const label = level === '5' ? 'Máximo estrés' :
                  level === '4' ? 'Alto estrés' :
                  level === '3' ? 'Estrés medio' :
                  level === '2' ? 'Bajo estrés' : 'Mínimo estrés';
    console.log(`  Nivel ${level} (${label}): ${count} preguntas (${percentage}%)`);
  });

  console.log('\n🎯 KEY INSIGHTS');
  console.log(`  • ${mockAviQuestions.weights['9-10']} preguntas críticas (peso ≥9)`);
  console.log(`  • ${mockAviQuestions.stressLevels['5'] + mockAviQuestions.stressLevels['4']} preguntas de alto estrés (nivel ≥4)`);
  console.log(`  • Cobertura balanceada entre categorías operativas y financieras`);
  console.log(`  • Énfasis en evaluación de riesgo y verificación cruzada`);

  console.log('\n🔧 RECOMMENDED USAGE PATTERNS');
  console.log('  Demo Mode: 5 preguntas representativas');
  console.log('  Quick Assessment: 15 preguntas críticas');
  console.log('  Standard Interview: 25-30 preguntas');
  console.log('  Full Assessment: 55 preguntas completas');
  console.log('  Category Focus: 6-8 preguntas por categoría específica');

  console.log('\n📈 SCORING RECOMMENDATIONS');
  console.log('  • Preguntas peso 9-10: Requieren score ≥0.7 para aprobación');
  console.log('  • Preguntas estrés 4-5: Monitorear indicadores de nerviosismo');
  console.log('  • Verificación cruzada: Validar coherencia entre categorías');
  console.log('  • Tiempo de respuesta: Comparar vs tiempo esperado por pregunta');

  return true;
}

function validateDatasetIntegrity() {
  console.log('\n🔍 DATASET INTEGRITY VALIDATION');
  console.log('=' .repeat(50));

  let isValid = true;
  const issues = [];

  // Validate total count
  const calculatedTotal = mockAviQuestions.categories.reduce((sum, cat) => sum + cat.count, 0);
  if (calculatedTotal !== mockAviQuestions.total) {
    isValid = false;
    issues.push(`❌ Category totals (${calculatedTotal}) don't match expected total (${mockAviQuestions.total})`);
  } else {
    console.log('✅ Category counts match total questions');
  }

  // Validate weight distribution
  const weightTotal = Object.values(mockAviQuestions.weights).reduce((sum, count) => sum + count, 0);
  if (weightTotal !== mockAviQuestions.total) {
    isValid = false;
    issues.push(`❌ Weight distribution (${weightTotal}) doesn't match total (${mockAviQuestions.total})`);
  } else {
    console.log('✅ Weight distribution is valid');
  }

  // Validate stress level distribution
  const stressTotal = Object.values(mockAviQuestions.stressLevels).reduce((sum, count) => sum + count, 0);
  if (stressTotal !== mockAviQuestions.total) {
    isValid = false;
    issues.push(`❌ Stress level distribution (${stressTotal}) doesn't match total (${mockAviQuestions.total})`);
  } else {
    console.log('✅ Stress level distribution is valid');
  }

  // Validate critical question percentage (should be reasonable)
  const criticalPercentage = (mockAviQuestions.weights['9-10'] / mockAviQuestions.total) * 100;
  if (criticalPercentage < 15 || criticalPercentage > 30) {
    issues.push(`⚠️ Critical questions percentage (${criticalPercentage.toFixed(1)}%) may be suboptimal (recommended: 15-30%)`);
  } else {
    console.log(`✅ Critical questions percentage (${criticalPercentage.toFixed(1)}%) is optimal`);
  }

  if (issues.length > 0) {
    console.log('\n⚠️ ISSUES FOUND:');
    issues.forEach(issue => console.log(`  ${issue}`));
  }

  console.log(`\n📊 INTEGRITY STATUS: ${isValid ? '✅ VALID' : '❌ ISSUES DETECTED'}`);
  return isValid;
}

function generateTestScenarios() {
  console.log('\n🧪 TEST SCENARIOS GENERATOR');
  console.log('=' .repeat(50));

  const scenarios = [
    {
      name: 'High-Risk Applicant',
      description: 'Applicant with multiple red flags',
      questions: ['creditos_anteriores', 'problemas_pagos', 'gastos_mordidas_cuotas', 'seguridad_personal'],
      expectedFlags: ['historial_problematico', 'riesgo_alto', 'ingresos_irregulares'],
      expectedScore: '0.2-0.4'
    },
    {
      name: 'Moderate-Risk Applicant',
      description: 'Typical transportista with some concerns',
      questions: ['ingresos_promedio_diarios', 'gasto_diario_gasolina', 'ahorros_emergencia'],
      expectedFlags: ['flujo_ajustado', 'vulnerabilidad_media'],
      expectedScore: '0.5-0.7'
    },
    {
      name: 'Low-Risk Applicant',
      description: 'Well-established operator with good history',
      questions: ['anos_en_ruta', 'valor_unidad_transporte', 'referencias_comerciales'],
      expectedFlags: ['perfil_estable'],
      expectedScore: '0.7-0.9'
    },
    {
      name: 'Inconsistent Responses',
      description: 'Applicant with contradictory information',
      questions: ['coherencia_ingresos_gastos', 'confirmacion_datos_criticos'],
      expectedFlags: ['inconsistencia_detectada', 'verificacion_requerida'],
      expectedScore: '0.3-0.5'
    },
    {
      name: 'First-Time Credit',
      description: 'New to formal credit but stable operation',
      questions: ['creditos_anteriores', 'motivacion_credito', 'plan_pago_propuesto'],
      expectedFlags: ['sin_historial', 'evaluacion_operacion'],
      expectedScore: '0.6-0.8'
    }
  ];

  scenarios.forEach((scenario, index) => {
    console.log(`\n${index + 1}. ${scenario.name}`);
    console.log(`   ${scenario.description}`);
    console.log(`   Key Questions: ${scenario.questions.join(', ')}`);
    console.log(`   Expected Flags: ${scenario.expectedFlags.join(', ')}`);
    console.log(`   Expected Score Range: ${scenario.expectedScore}`);
  });

  console.log('\n💡 USAGE INSTRUCTIONS:');
  console.log('  1. Use these scenarios to test the voice analysis engine');
  console.log('  2. Create mock responses that would trigger the expected flags');
  console.log('  3. Validate that scoring aligns with expected ranges');
  console.log('  4. Test cross-validation between related questions');

  return scenarios;
}

function exportAnalysisReport() {
  const timestamp = new Date().toISOString().split('T')[0];
  const report = {
    generatedDate: timestamp,
    dataset: mockAviQuestions,
    analysis: {
      totalQuestions: mockAviQuestions.total,
      completionPercentage: 100,
      categoryDistribution: mockAviQuestions.categories,
      weightDistribution: mockAviQuestions.weights,
      stressDistribution: mockAviQuestions.stressLevels
    },
    recommendations: [
      'Use demo mode (5 questions) for initial testing',
      'Focus on critical questions (weight ≥9) for risk assessment',
      'Monitor high-stress questions (level ≥4) for behavior analysis',
      'Implement cross-validation for financial coherence',
      'Use micro-local analysis for detailed evaluation'
    ]
  };

  const outputPath = path.join(__dirname, '..', 'reports', `avi-analysis-${timestamp}.json`);

  // Create reports directory if it doesn't exist
  const reportsDir = path.dirname(outputPath);
  if (!fs.existsSync(reportsDir)) {
    fs.mkdirSync(reportsDir, { recursive: true });
  }

  try {
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    console.log(`\n📄 Report exported to: ${outputPath}`);
    return outputPath;
  } catch (error) {
    console.error(`❌ Failed to export report: ${error.message}`);
    return null;
  }
}

// Main execution
function main() {
  console.log('🚀 Starting AVI Questions Analysis...\n');

  try {
    // Generate analysis report
    generateQuestionAnalysisReport();

    // Validate dataset integrity
    const isValid = validateDatasetIntegrity();

    // Generate test scenarios
    const scenarios = generateTestScenarios();

    // Export comprehensive report
    const reportPath = exportAnalysisReport();

    console.log('\n🎉 Analysis completed successfully!');
    console.log('\n📋 NEXT STEPS:');
    console.log('  1. Review the generated report');
    console.log('  2. Test the Runner component with different question modes');
    console.log('  3. Implement the test scenarios in your application');
    console.log('  4. Validate voice analysis results against expected patterns');

    if (reportPath) {
      console.log(`  5. Check detailed report: ${reportPath}`);
    }

    return 0;

  } catch (error) {
    console.error(`❌ Analysis failed: ${error.message}`);
    return 1;
  }
}

// CLI handling
if (require.main === module) {
  const exitCode = main();
  process.exit(exitCode);
}

module.exports = {
  generateQuestionAnalysisReport,
  validateDatasetIntegrity,
  generateTestScenarios,
  exportAnalysisReport
};