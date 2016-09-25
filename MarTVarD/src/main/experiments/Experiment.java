package main.experiments;

import main.distance.Distance;
import main.distance.TotalVariation;
import main.models.NaiveMatrix;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public abstract class Experiment {
    Map<int[], ArrayList<ExperimentResult>> resultMap;
    int nAttributesActive;
    Instance sampleInstance;
    Distance distanceMetric = new TotalVariation();
    int[] attributeIndices;
    int[] classIndices;
    abstract ArrayList<ExperimentResult> getResults(NaiveMatrix model1, NaiveMatrix model2, Instances allInstances);

    public Experiment(Instances instances1, Instances instances2, int nAttributesActive, int[] attributeIndices, int[] classIndices) {

        // Generate base models for each data set
        NaiveMatrix model1 = new NaiveMatrix(instances1);
        NaiveMatrix model2 = new NaiveMatrix(instances2);

        // Generate union set of all instances in both data sets
        Instances allInstances = new Instances(instances1);
        allInstances.addAll(instances2);

        // Store needed metadata
        this.sampleInstance = allInstances.firstInstance();
        this.nAttributesActive = nAttributesActive;

        // Get number of combinations from choosing nAttributesActive of attributes from instances there are
        int nCombination = nCr(attributeIndices.length, nAttributesActive);
        resultMap = new HashMap<>();
        for (int i = 0; i < nCombination; i++) {
            System.out.print("\rRunning experiment " + (i + 1) + "/" + nCombination);
            // Get combination between attributes
            int[] indices = getKthCombination(i, attributeIndices, nAttributesActive);
            indices = ArrayUtils.addAll(indices, classIndices);
            resultMap.put(indices, getResults(model1, model2, generateAllPartialInstances(allInstances, indices)));
        }
        System.out.print("\n");
        this.resultMap = sortByValue(this.resultMap);
        this.attributeIndices = attributeIndices;
        this.classIndices = classIndices;
    }

    public String[][] getResultTable() {
        return this.getResultTable(0, "*");
    }

    public String[][] getResultTable(int classIndex, String className) {
        int[][] attributeSubSets = this.resultMap.keySet().toArray(new int[this.resultMap.size()][this.nAttributesActive]);
       String[][] results = new String[attributeSubSets.length][9];
        for (int i = 0; i < attributeSubSets.length; i++) {
            ExperimentResult currentResult = this.resultMap.get(attributeSubSets[i]).get(classIndex);
            results[i][0] = Double.toString(currentResult.actualResult);
            results[i][1] = Double.toString(currentResult.mean);
            results[i][2] = Double.toString(currentResult.sd);
            results[i][3] = Double.toString(currentResult.maxDist);
            results[i][4] = "";
            results[i][5] = Double.toString(currentResult.minDist);
            results[i][6] = "";
            results[i][7] = "";
            for (int j = 0; j < attributeSubSets[i].length; j++) {
                results[i][7] += this.sampleInstance.attribute(attributeSubSets[i][j]).name() + "_";
            }
            results[i][7] = results[i][7].substring(0, results[i][7].length() - 1);
            results[i][8] = className;
            if (!Double.isInfinite(currentResult.actualResult)) {
                for (int j = 0; j < attributeSubSets[i].length; j++) {
                    int attributeIndex = attributeSubSets[i][j];
                    String minVal = Double.isNaN(currentResult.minValues[attributeIndex]) || (int)currentResult.minValues[attributeIndex] < 0 ? "*" :
                            this.sampleInstance.attribute(attributeIndex).value((int)currentResult.minValues[attributeIndex]);
                    String maxVal = Double.isNaN(currentResult.minValues[attributeIndex]) || (int)currentResult.maxValues[attributeIndex] < 0 ? "*" :
                            this.sampleInstance.attribute(attributeIndex).value((int)currentResult.maxValues[attributeIndex]);
                    results[i][4] += this.sampleInstance.attribute(attributeIndex).name() + "=" + maxVal + "_";
                    results[i][6] += this.sampleInstance.attribute(attributeIndex).name() + "=" + minVal + "_";
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

    private static Instances generateAllPartialInstances(Instances instances, int[] attributesIndices) {
        Instances partialInstances = new Instances(instances, instances.size());
        HashSet<Integer> existingPartialInstances = new HashSet<>();
        for (int i = 0; i < instances.size(); i++) {
            Instance instance = new DenseInstance(instances.instance(i));
            int partialHash = 0;
            int hashBase = 1;
            // Iterate over attributes in instance
            for (int j = 0; j < instances.numAttributes(); j ++) {
                // If not class or active attribute set missing
                if (!ArrayUtils.contains(attributesIndices, j)) {
                    instance.setMissing(j);
                }
                // Else calculate hash of attribute value and add to total partial instance hash
                else {
                    partialHash += hashBase * instance.value(j);
                    hashBase *= instances.attribute(j).numValues();
                }
            }
            // Check if partial Instance already exists in data set
            // If true, delete duplicate instance from data set
            // Else add partial instance hash to set
            if (!existingPartialInstances.contains(partialHash)) {
                partialInstances.add(instance);
                existingPartialInstances.add(partialHash);
            }
        }
        return partialInstances;
    }

    private static Instances generateSamplePartialInstances(Instances instances, int[] attributeIndices, int sampleSize) {
        Instances sampleInstances = new Instances(instances, sampleSize);
        HashSet<Integer> selectedInstances = new HashSet<>();
        Random rng = new Random();
        while (sampleInstances.size() < sampleSize) {
            int index = rng.nextInt(instances.size());
            if (!selectedInstances.contains(index)) {
                selectedInstances.add(index);
                sampleInstances.add(instances.get(index));
            }
        }
        return generateAllPartialInstances(sampleInstances, attributeIndices);
    }

    private static Instances separateInstanceAttributes(Instances instances, int[] attributesIndices) {
        // Create new Instances
        String name = "";
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int index : attributesIndices) {
            name += "v"+index;
            attributes.add(new Attribute("v"+index,
                    getAttributeValues(instances, index),
                    instances.attribute(index).getMetadata()));
        }
        attributes.add(new Attribute("class",
                getAttributeValues(instances, instances.classIndex()),
                instances.attribute(instances.classIndex()).getMetadata()));
        Instances newInstances = new Instances(name, attributes, instances.size());
        newInstances.setClassIndex(attributes.size() - 1);

        // Copy value of certain attributes and class from old to new instances
        for (Instance instance : instances) {
            double[] inst = new double[attributes.size()];
            for (int i = 0; i < attributesIndices.length; i++) {
                inst[i] = instance.value(attributesIndices[i]);
            }
            inst[inst.length - 1] = instance.classValue();
            Instance newInstance = new DenseInstance(1, inst);
            newInstance.setDataset(newInstances);
            newInstances.add(newInstance);
        }

        return newInstances;
    }

    private static ArrayList<String> getAttributeValues(Instances instances, int attributeIndex) {
        ArrayList<String> values = new ArrayList<>();
        for (int j = 0; j < instances.attribute(attributeIndex).numValues(); j++) {
            values.add(instances.attribute(attributeIndex).value(j));
        }
        return values;
    }

    private static int[] getKthCombination(int k, int[] elements, int choices) {
        if (choices == 0) return new int[]{};
        else if (elements.length == choices) return  elements;
        else {
            int nCombinations = nCr(elements.length - 1, choices - 1);
            if (k < nCombinations) return ArrayUtils.addAll(ArrayUtils.subarray(elements, 0, 1),
                    getKthCombination(k, ArrayUtils.subarray(elements, 1, elements.length), choices - 1));
            else return getKthCombination(k - nCombinations, ArrayUtils.subarray(elements, 1, elements.length), choices);
        }
    }

    private static int nCr(int n, int r) {
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
        Collections.sort( list, new Comparator<Map.Entry<int[], ArrayList<ExperimentResult>>>() {
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
