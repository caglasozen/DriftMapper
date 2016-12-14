package main.report;

import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instances;

import java.math.BigInteger;
import java.util.*;

/**
 * Created by loongkuan on 6/12/2016.
 */
public class StructuredExperimentResult extends ExperimentResult{
    Map<BigInteger, ExperimentResult> separateExperiments;
    Map<BigInteger, Double> experimentsProbability;
    int[] conditionedAttributes;

    public StructuredExperimentResult(double weightAverageDistance, double[] separateDistance, Instances instances,
                                      Map<BigInteger, ExperimentResult> separateExperiments,
                                      Map<BigInteger, Double> experimentsProbability, int[] conditionedAttributes) {
        super(weightAverageDistance, separateDistance, instances);
        this.separateExperiments = separateExperiments;
        this.experimentsProbability = experimentsProbability;
        this.conditionedAttributes = conditionedAttributes;
    }

    StructuredExperimentResult(ArrayList<ExperimentResult> resultList) {
        super(resultList);
        // Initialise needed places to put data
        Map<BigInteger , Double> groupedExperimentProbabilitySum = new HashMap<>();
        Map<BigInteger, ArrayList<ExperimentResult>> groupedSeparateExperiments = new HashMap<>();
        // Iterate through all the given structured experiment results to average
        for (ExperimentResult result : resultList) {
            Map<BigInteger, ExperimentResult> currentSeparateExperiments = ((StructuredExperimentResult)result).separateExperiments;
            Map<BigInteger, Double> currentExperimentsProbability = ((StructuredExperimentResult)result).experimentsProbability;
            for (BigInteger hash : currentSeparateExperiments.keySet()) {
                // Check if we seen this before
                if (!groupedSeparateExperiments.containsKey(hash) || !groupedExperimentProbabilitySum.containsKey(hash)) {
                    groupedSeparateExperiments.put(hash, new ArrayList<>());
                    groupedExperimentProbabilitySum.put(hash, 0.0);
                }
                // Add the current experiment/probability to the group it belongs to
                ArrayList<ExperimentResult> tmpGroup = groupedSeparateExperiments.get(hash);
                tmpGroup.add(currentSeparateExperiments.get(hash));
                groupedSeparateExperiments.put(hash, tmpGroup);

                Double tmpProb = groupedExperimentProbabilitySum.get(hash);
                tmpProb += currentExperimentsProbability.get(hash);
                groupedExperimentProbabilitySum.put(hash, tmpProb);
            }
        }

        // Initialise object's separateExperiments and experimentsProbability
        this.separateExperiments = new HashMap<>();
        this.experimentsProbability = new HashMap<>();
        // Iterate though all the "groups" seen
        for (BigInteger hash : groupedSeparateExperiments.keySet()) {
            ArrayList<ExperimentResult> groupResults = groupedSeparateExperiments.get(hash);
            Double groupProbSum = groupedExperimentProbabilitySum.get(hash);
            int groupLength = groupResults.size();
            this.experimentsProbability.put(hash, groupProbSum / (double)groupLength);
            this.separateExperiments.put(hash, new SingleExperimentResult(groupResults));
        }
        this.conditionedAttributes = ((StructuredExperimentResult)resultList.get(0)).conditionedAttributes;
    }

    public String[][] getDetailedSubTable() {
        ArrayList<String[]> subtable = new ArrayList<>();
        for(BigInteger hash : this.separateExperiments.keySet()) {
            ExperimentResult result = this.separateExperiments.get(hash);
            String resultRow[] = new String[10];

            resultRow[0] = "";
            for (int i = 0; i < result.instances.get(0).numAttributes(); i++) {
                if (!result.instances.get(0).isMissing(i)) {
                    resultRow[0] += result.instances.attribute(i).name() + "_";
                }
            }
            resultRow[0] = resultRow[0].substring(0, resultRow[0].length() - 1);

            resultRow[1] = "";
            for (int i: this.conditionedAttributes) {
                resultRow[1] += result.instances.attribute(i).name() + "=" +
                        result.instances.get(0).stringValue(i) + "_";
            }
            resultRow[1] = resultRow[1].substring(0, resultRow[1].length() - 1);

            resultRow[2] = Double.toString(this.experimentsProbability.get(hash));

            resultRow[3] = Double.toString(result.distance);
            resultRow[4] = Double.toString(result.mean);
            resultRow[5] = Double.toString(result.sd);
            resultRow[6] = Double.toString(result.maxDist);
            resultRow[7] = "";
            resultRow[8] = Double.toString(result.minDist);
            resultRow[9] = "";

            if (!Double.isInfinite(result.distance)) {
                for (int j = 0; j < result.instances.numAttributes(); j++) {
                    if (!result.maxInstance.isMissing(j) && !result.minInstance.isMissing(j)) {
                        if (!ArrayUtils.contains(this.conditionedAttributes, j)) {
                            String maxVal = result.maxInstance.stringValue(j);
                            resultRow[7] += result.instances.attribute(j).name() + "=" + maxVal + "_";
                            String minVal = result.minInstance.stringValue(j);
                            resultRow[9] += result.instances.attribute(j).name() + "=" + minVal + "_";
                        }
                    }
                }
                // Trim last underscore
                resultRow[7] = resultRow[7].substring(0, resultRow[7].length() - 1);
                resultRow[9] = resultRow[9].substring(0, resultRow[9].length() - 1);
            }
            else {
                resultRow[7] = "NA";
                resultRow[9] = "NA";
            }
            subtable.add(resultRow);
        }
        return subtable.toArray(new String[subtable.size()][subtable.get(0).length]);
    }
}
