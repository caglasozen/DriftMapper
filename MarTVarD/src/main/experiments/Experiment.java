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

/**
 * Created by LoongKuan on 31/07/2016.
 **/
public abstract class Experiment {
    Map<int[], ArrayList<ExperimentResult>> resultMap;
    int nAttributesActive;
    Instance sampleInstance;
    Distance distanceMetric = new TotalVariation();
    abstract ArrayList<ExperimentResult> getResults(NaiveMatrix model1, NaiveMatrix model2, Instances allInstances);

    public Experiment(Instances instances1, Instances instances2, int nAttributesActive) {
        // List of 0 to n where n is the number of attributes
        int[] attributeIndices = new int[instances1.numAttributes() - 1];
        for (int i = 0; i < instances1.numAttributes() - 1; i++) attributeIndices[i] = i;

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
        int nCombination = nCr(instances1.numAttributes() - 1, nAttributesActive);
        resultMap = new HashMap<>();
        for (int i = 0; i < nCombination; i++) {
            System.out.print("\rRunning experiment " + (i + 1) + "/" + nCombination);
            // Get combination between attributes
            int[] indices = getKthCombination(i, attributeIndices, nAttributesActive);
            resultMap.put(indices, getResults(model1, model2, generatePartialInstances(allInstances, indices)));
        }
        System.out.print("\n");
        this.resultMap = sortByValue(this.resultMap);
    }

    public String[][] getResultTable() {
        return this.getResultTable(0);
    }

    public String[][] getResultTable(int classIndex) {
        int[][] attributeSubSets = this.resultMap.keySet().toArray(new int[this.resultMap.size()][this.nAttributesActive]);
       String[][] results = new String[attributeSubSets.length][8];
        for (int i = 0; i < attributeSubSets.length; i++) {
            ExperimentResult currentResult = this.resultMap.get(attributeSubSets[i]).get(classIndex);
            results[i][0] = Double.toString(currentResult.actualResult);
            results[i][1] = Double.toString(currentResult.mean);
            results[i][2] = Double.toString(currentResult.sd);
            results[i][3] = Double.toString(currentResult.minDist);
            results[i][4] = "";
            results[i][5] = Double.toString(currentResult.maxDist);
            results[i][6] = "";
            results[i][7] = "";
            if (!Double.isInfinite(currentResult.actualResult)) {
                for (int j = 0; j < attributeSubSets[i].length; j++) {
                    String minVal = Double.isNaN(currentResult.minValues[j]) || (int)currentResult.minValues[j] < 0 ? "*" :
                            this.sampleInstance.attribute(attributeSubSets[i][j]).value((int)currentResult.minValues[attributeSubSets[i][j]]);
                    String maxVal = Double.isNaN(currentResult.minValues[j]) || (int)currentResult.maxValues[j] < 0 ? "*" :
                            this.sampleInstance.attribute(attributeSubSets[i][j]).value((int)currentResult.maxValues[attributeSubSets[i][j]]);
                    results[i][4] += "_" + this.sampleInstance.attribute(attributeSubSets[i][j]).name() + "=" + maxVal;
                    results[i][6] += "_" + this.sampleInstance.attribute(attributeSubSets[i][j]).name() + "=" + minVal;
                    results[i][7] += "_" + this.sampleInstance.attribute(attributeSubSets[i][j]).name();
                }
                // Add class info
                String minVal = (int)currentResult.minValues[this.sampleInstance.classIndex()] < 0 ? "*" :
                        this.sampleInstance.attribute(this.sampleInstance.classIndex()).value((int)currentResult.minValues[this.sampleInstance.classIndex()]);
                String maxVal = (int)currentResult.maxValues[this.sampleInstance.classIndex()] < 0 ? "*" :
                        this.sampleInstance.attribute(this.sampleInstance.classIndex()).value((int)currentResult.maxValues[this.sampleInstance.classIndex()]);
                results[i][4] += "_" + this.sampleInstance.attribute(this.sampleInstance.classIndex()).name() + "=" + maxVal;
                results[i][6] += "_" + this.sampleInstance.attribute(this.sampleInstance.classIndex()).name() + "=" + minVal;
            }
            else {
                results[i][4] = "Does not exist";
                results[i][6] = "Does not exist";
                results[i][7] = "Does not exist";
            }
        }
        return results;
    }

    private static Instances generatePartialInstances(Instances instances, int[] attributesIndices) {
        Instances partialInstances = new Instances(instances, instances.size());
        HashSet<Integer> existingPartialInstances = new HashSet<>();
        for (int i = 0; i < instances.size(); i++) {
            Instance instance = new DenseInstance(instances.instance(i));
            int partialHash = 0;
            int hashBase = 1;
            // Iterate over attributes in instance
            for (int j = 0; j < instances.numAttributes(); j ++) {
                // If not class or active attribute set missing
                if (instances.classIndex() != j && !ArrayUtils.contains(attributesIndices, j)) {
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
