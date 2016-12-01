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
    protected int[] attributesAvailable;
    protected int attributeSubsetLength;
    protected Instances allInstances;

    protected Distance distanceMetric = new TotalVariation();

    public abstract Model copy();
    public abstract void reset();

    public abstract ExperimentResult findCovariateDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);
    public abstract ExperimentResult findJointDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);
    public abstract ExperimentResult findLikelihoodDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);
    public abstract ExperimentResult findPosteriorDistance(
            Model modelToCompare, int[] attributeSubset, double sampleScale);

    public abstract void addInstance(Instance instance);
    public abstract void removeInstance(Instance instance);

    public int size() {
        return this.allInstances.size();
    }

    public int[] getAttributesAvailable() {
        return attributesAvailable;
    }

    public void addAll(Instances instances) {
        int percentage = -1;
        for (int i = 0; i < instances.size(); i++) {
            if (percentage != (int)((i/(double)instances.size()) * 100)) {
                percentage = (int)((i/(double)instances.size()) * 100);
                System.out.print("\rAdded " + percentage + "% of Instances ");
            }
            this.addInstance(instances.get(i));
        }
    }

    public void removeAll(Instances instances) {
        for (Instance instance : instances) {
            this.removeInstance(instance);
        }
    }

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

    public Map<int[], ExperimentResult> analyseDifference(Model modelToCompare, double sampleScale,
                                                          int nTests, DriftMeasurement driftMeasurement) {
        Map<int[], ExperimentResult> resultMap = new HashMap<>();
        int nCombination = nCr(this.attributesAvailable.length, this.attributeSubsetLength);

        for (int i = 0; i < nCombination; i++) {
            System.out.print("\rRunning experiment " + (i + 1) + "/" + nCombination);
            // Get attribute subset
            int[] attributeSubset = getKthCombination(i, this.attributesAvailable, this.attributeSubsetLength);
            ArrayList<ExperimentResult> results = new ArrayList<>();
            for (int j = 0; j < nTests; j++) {
                results.add(getResults(modelToCompare, attributeSubset, sampleScale, driftMeasurement));
            }
            resultMap.put(attributeSubset, new ExperimentResult(results));
        }
        System.out.print("\n");
        resultMap = sortByValue(resultMap);
        return resultMap;
    }

    private ExperimentResult getResults(Model modelToCompare, int[] attributeSubset,
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
        return null;
    }

    public String[][] getResultTable(Map<int[], ExperimentResult> resultMap) {
        int[][] attributeSubSets = resultMap.keySet().toArray(new int[resultMap.size()][this.attributeSubsetLength]);
        String[][] results = new String[attributeSubSets.length][8];
        for (int i = 0; i < attributeSubSets.length; i++) {
            ExperimentResult currentResult = resultMap.get(attributeSubSets[i]);
            results[i][0] = Double.toString(currentResult.actualResult);
            results[i][1] = Double.toString(currentResult.mean);
            results[i][2] = Double.toString(currentResult.sd);
            results[i][3] = Double.toString(currentResult.maxDist);
            results[i][4] = "";
            results[i][5] = Double.toString(currentResult.minDist);
            results[i][6] = "";
            results[i][7] = "";
            for (int j = 0; j < attributeSubSets[i].length; j++) {
                results[i][7] += this.allInstances.attribute(attributeSubSets[i][j]).name() + "_";
            }
            results[i][7] = results[i][7].substring(0, results[i][7].length() - 1);
            if (!Double.isInfinite(currentResult.actualResult)) {
                for (int j = 0; j < this.allInstances.numAttributes(); j++) {
                    if (!Double.isNaN(currentResult.maxValues[j]) && !Double.isNaN(currentResult.minValues[j])) {
                        String maxVal = this.allInstances.attribute(j).value((int)currentResult.maxValues[j]);
                        results[i][4] += this.allInstances.attribute(j).name() + "=" + maxVal + "_";

                        String minVal = this.allInstances.attribute(j).value((int)currentResult.minValues[j]);
                        results[i][6] += this.allInstances.attribute(j).name() + "=" + minVal + "_";
                    }
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

    private static Map<int[], ExperimentResult> sortByValue( Map<int[], ExperimentResult> map ) {
        List<Map.Entry<int[], ExperimentResult>> list = new LinkedList<>(map.entrySet());
        list.sort( new Comparator<Map.Entry<int[], ExperimentResult>>() {
            public int compare( Map.Entry<int[], ExperimentResult> o1, Map.Entry<int[], ExperimentResult> o2 )
            {
                double value = o1.getValue().actualResult - o2.getValue().actualResult;
                if (value == 0.0f) return 0;
                else if(value < 0.0f) return -1;
                else return 1;
            }
        } );

        Map<int[], ExperimentResult> result = new LinkedHashMap<>();
        for (Map.Entry<int[], ExperimentResult> entry : list)
        {
            result.put( entry.getKey(), entry.getValue() );
        }
        return result;
    }
}
