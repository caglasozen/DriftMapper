package main.models;

import main.distance.Distance;
import main.distance.TotalVariation;
import main.analyse.ExperimentResult;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 28/11/2016.
 **/

public abstract class Model {
    // TODO: get rif of total freq
    protected int totalFrequency;
    // TODO: get rid of att avail
    protected int[] attributesAvailable;
    protected int nAttributesActive;
    protected Instance exampleInst;
    protected Instances allInstances;

    protected Distance distanceMetric = new TotalVariation();

    public abstract ArrayList<ExperimentResult> findCovariateDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);
    public abstract ArrayList<ExperimentResult> findJointDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);
    public abstract ArrayList<ExperimentResult> findLikelihoodDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);
    public abstract ArrayList<ExperimentResult> findPosteriorDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);

    public abstract void addInstance(Instance instance);
    public abstract void removeInstance(Instance instance);

    protected static Instances sampleInstances(Instances instances, double sampleScale) {
        Instances sampleInstances = new Instances(instances, (int)(instances.size() * sampleScale));
        HashSet<Integer> selectedInstances = new HashSet<>();
        Random rng = new Random();
        if (sampleScale >= 1.0f || sampleScale < 0.0f) {
            sampleInstances = instances;
        }
        else if (sampleScale == 0.0f) {
            return sampleInstances;
        }
        else {
            do {
                int index = rng.nextInt(instances.size());
                if (!selectedInstances.contains(index)) {
                    selectedInstances.add(index);
                    sampleInstances.add(instances.get(index));
                }
            } while (sampleInstances.size() < (int)(instances.size() / sampleScale));
        }
        return sampleInstances;
    }

    protected static int[] getKthCombination(int k, int[] elements, int choices) {
        if (choices == 0) return new int[]{};
        else if (elements.length == choices) return  elements;
        else {
            int nCombinations = nCr(elements.length - 1, choices - 1);
            if (k < nCombinations) return ArrayUtils.addAll(ArrayUtils.subarray(elements, 0, 1),
                    getKthCombination(k, ArrayUtils.subarray(elements, 1, elements.length), choices - 1));
            else return getKthCombination(k - nCombinations, ArrayUtils.subarray(elements, 1, elements.length), choices);
        }
    }

    public Map<int[], ArrayList<ExperimentResult>> analyseDifference(Model modelToCompare, double sampleScale,
                                                                     int nTests, DriftMeasurement driftMeasurement) {

        Map<int[], ArrayList<ExperimentResult>> resultMap = new HashMap<>();
        int nCombination = nCr(this.attributesAvailable.length, this.nAttributesActive);

        for (int i = 0; i < nCombination; i++) {
            System.out.print("\rRunning experiment " + (i + 1) + "/" + nCombination);
            // Get attribute subset
            int[] attributeSubset = getKthCombination(i, this.attributesAvailable, this.nAttributesActive);

            ArrayList<ArrayList<ExperimentResult>> results = new ArrayList<>();
            for (int j = 0; j < nTests; j++) {
                ArrayList<ExperimentResult> tmpRes = getResults(modelToCompare, attributeSubset, sampleScale, driftMeasurement);
                for (int k = 0; k < tmpRes.size(); k++) {
                    if (results.size() <= k) results.add(new ArrayList<>());
                    results.get(k).add(tmpRes.get(k));
                }
            }
            ArrayList<ExperimentResult> finalAveragedResults = new ArrayList<>();
            for (int k = 0; k < results.size(); k++) {
                if (!results.get(k).isEmpty()) {
                    finalAveragedResults.add(new ExperimentResult(results.get(k)));
                }
            }
            resultMap.put(attributeSubset, finalAveragedResults);
        }
        System.out.print("\n");
        resultMap = sortByValue(resultMap);
        return resultMap;
    }

    private ArrayList<ExperimentResult> getResults(Model modelToCompare, int[] attributeSubset,
                                                   double sampleScale, DriftMeasurement driftMeasurement) {
        switch (driftMeasurement) {
            case COVARIATE:
                return this.findCovariateDistance(modelToCompare, attributeSubset, sampleScale);
            case JOINT:
                return this.findJointDistance(modelToCompare, attributeSubset, sampleScale);
            case LIKELIHOOD:
                return this.findLikelihoodDistance(modelToCompare, attributeSubset, sampleScale);
            case POSTERIOR:
                return this.findPosteriorDistance(modelToCompare, attributeSubset, sampleScale);
        }
        return new ArrayList<>();
    }

    public String[][] getResultTable(DriftMeasurement driftMeasurement, Map<int[], ArrayList<ExperimentResult>> resultMap) {
        switch (driftMeasurement) {
            case COVARIATE:
                return this.getResultTable(0, "NA", resultMap);
            case JOINT:
                //TODO: Fix labelling for joint
                return this.getResultTable(0, "NA", resultMap);
            case LIKELIHOOD:
                int nCombinations = resultMap.size();
                int nClasses = this.exampleInst.numClasses();
                String[][] collatedResults = new String[nCombinations*nClasses][9];
                for (int i = 0; i < nClasses; i++) {
                    String[][] tmpResult = this.getResultTable(i, this.exampleInst.classAttribute().value(i), resultMap);
                    for (int j = 0; j < tmpResult.length; j++) {
                        collatedResults[i*nCombinations + j] = tmpResult[j];
                    }
                }
                return collatedResults;
            case POSTERIOR:
                return this.getResultTable(0, "*", resultMap);
        }
        return this.getResultTable(0, "*", resultMap);
    }

    private String[][] getResultTable(int classIndex, String className, Map<int[], ArrayList<ExperimentResult>> resultMap) {
        int[][] attributeSubSets = resultMap.keySet().toArray(new int[resultMap.size()][this.nAttributesActive]);
        String[][] results = new String[attributeSubSets.length][9];
        for (int i = 0; i < attributeSubSets.length; i++) {
            ExperimentResult currentResult = resultMap.get(attributeSubSets[i]).get(classIndex);
            results[i][0] = Double.toString(currentResult.actualResult);
            results[i][1] = Double.toString(currentResult.mean);
            results[i][2] = Double.toString(currentResult.sd);
            results[i][3] = Double.toString(currentResult.maxDist);
            results[i][4] = "";
            results[i][5] = Double.toString(currentResult.minDist);
            results[i][6] = "";
            results[i][7] = "";
            for (int j = 0; j < attributeSubSets[i].length; j++) {
                results[i][7] += this.exampleInst.attribute(attributeSubSets[i][j]).name() + "_";
            }
            results[i][7] = results[i][7].substring(0, results[i][7].length() - 1);
            results[i][8] = className;
            if (!Double.isInfinite(currentResult.actualResult)) {
                for (int j = 0; j < attributeSubSets[i].length; j++) {
                    int attributeIndex = attributeSubSets[i][j];
                    String minVal = Double.isNaN(currentResult.minValues[attributeIndex]) || (int)currentResult.minValues[attributeIndex] < 0 ? "*" :
                            this.exampleInst.attribute(attributeIndex).value((int)currentResult.minValues[attributeIndex]);
                    String maxVal = Double.isNaN(currentResult.minValues[attributeIndex]) || (int)currentResult.maxValues[attributeIndex] < 0 ? "*" :
                            this.exampleInst.attribute(attributeIndex).value((int)currentResult.maxValues[attributeIndex]);
                    results[i][4] += this.exampleInst.attribute(attributeIndex).name() + "=" + maxVal + "_";
                    results[i][6] += this.exampleInst.attribute(attributeIndex).name() + "=" + minVal + "_";
                }
                // Trim last underscore
                results[i][4] = results[i][4].substring(0, results[i][4].length() - 1);
                results[i][6] = results[i][6].substring(0, results[i][6].length() - 1);
            }
            else {
                results[i][4] = "NA";
                results[i][6] = "NA";
            }
        }
        return results;
    }

    protected static int nCr(int n, int r) {
        if (r >= n /2) r = n - r;
        int ans = 1;
        for (int i = 1; i <= r; i++) {
            ans *= n - r + i;
            ans /= i;
        }
        return ans;
    }

    private static Map<int[], ArrayList<ExperimentResult>> sortByValue( Map<int[], ArrayList<ExperimentResult>> map ) {
        List<Map.Entry<int[], ArrayList<ExperimentResult>>> list = new LinkedList<>(map.entrySet());
        list.sort( new Comparator<Map.Entry<int[], ArrayList<ExperimentResult>>>() {
            public int compare( Map.Entry<int[], ArrayList<ExperimentResult>> o1, Map.Entry<int[], ArrayList<ExperimentResult>> o2 )
            {
                double value = o1.getValue().get(0).actualResult - o2.getValue().get(0).actualResult;
                if (value == 0.0f) return 0;
                else if(value < 0.0f) return -1;
                else return 1;
            }
        } );

        Map<int[], ArrayList<ExperimentResult>> result = new LinkedHashMap<>();
        for (Map.Entry<int[], ArrayList<ExperimentResult>> entry : list)
        {
            result.put( entry.getKey(), entry.getValue() );
        }
        return result;
    }
}
