package main.models;

import main.distance.Distance;
import main.distance.TotalVariation;
import main.experiments.ExperimentResult;
import org.apache.commons.lang3.ArrayUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;

/**
 * Created by loongkuan on 28/11/2016.
 **/

public abstract class Model {
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

    protected static Instances generatePartialInstances(Instances instances, int[] attributesIndices) {
        Instances partialInstances = new Instances(instances, instances.size());
        HashSet<String> existingPartialInstances = new HashSet<>();
        for (int i = 0; i < instances.size(); i++) {
            Instance instance = new DenseInstance(instances.instance(i));
            // Iterate over attributes in instance
            for (int j = 0; j < instances.numAttributes(); j ++) {
                // If not class or active attribute set missing
                if (!ArrayUtils.contains(attributesIndices, j)) {
                    instance.setMissing(j);
                }
            }
            // Check if partial Instance already exists in data set
            // If true, delete duplicate instance from data set
            // Else add partial instance hash to set
            if (!existingPartialInstances.contains(instance.toString())) {
                partialInstances.add(instance);
                existingPartialInstances.add(instance.toString());
            }
        }
        return partialInstances;
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
            } while (sampleInstances.size() < (int)(instances.size() * sampleScale));
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

    protected static int nCr(int n, int r) {
        if (r >= n /2) r = n - r;
        int ans = 1;
        for (int i = 1; i <= r; i++) {
            ans *= n - r + i;
            ans /= i;
        }
        return ans;
    }
}
