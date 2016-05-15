package main.models.sampling;

import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by Lee on 19/04/2016.
 **/
public abstract class AbstractSampler {
    protected Instances dataSet;
    double magnitudeScale = 0.0f;
    ArrayList<ArrayList<String>> allPossibleValues;
    Instances sampledInstances;
    int nInstances;
    int nInstancesGeneratedSoFar;

    public abstract Instances generateAllInstances();
    public abstract Instance nextInstance();
    public abstract AbstractSampler copy();
    public abstract void reset();

    public int getNInstances() {
        return this.nInstances;
    }

    public void setDataSet(Instances dataSet) {
        this.dataSet = dataSet;
        generateAllPossibleValues();

        int nCombinations = 1;
        for (ArrayList<String> attribute : this.allPossibleValues) {
            nCombinations *= attribute.size();
        }
        this.magnitudeScale = (double)nCombinations / (double)this.nInstances;
    }


    public double getMagnitudeScale() {
        return this.magnitudeScale;
    }

    public Instances getDataSet() {
        return dataSet;
    }

    public Instances getSampledInstances(){
        return sampledInstances;
    }

    public ArrayList<ArrayList<String>> getAllPossibleValues() {
        return allPossibleValues;
    }

    protected void generateAllPossibleValues() {
        allPossibleValues = new ArrayList<>();
        // NOTE: j represents x_n
        for (int j = 0; j < dataSet.numAttributes() - 1; j++) {
            // Initialise local variables for this loop (aka. data for this X variable)
            HashSet<String> possibleValues = new HashSet<>();
            for (int i = 0; i < dataSet.size(); i++) {
                String curKey = Double.toString(dataSet.instance(i).value(j));
                if (!possibleValues.contains(curKey)) possibleValues.add(curKey);
            }
            // Add all the keys/Possible values to allPossibleValues
            allPossibleValues.add(new ArrayList<>(possibleValues));
        }
        //Suggest Garbage collector to run
        System.gc();
    }

    public static Instances findIntersectionBetweenInstances(Instances instances1, Instances instances2) {
        // Create a hashed mapped set of instances1 first
        HashMap<Integer, Instance> baseMap = new HashMap<>(instances1.size());
        for (Instance instance : instances1) {
            Integer hash = Arrays.hashCode(instance.toDoubleArray());
            if (!baseMap.containsKey(hash)) baseMap.put(hash, instance);
        }

        // Check each instance in instances2, if it is in instances1's baseMap, put into final Instances to return
        Instances finalInstances = new Instances(instances2, instances1.size() + instances2.size());
        for (Instance instance : instances2) {
            Integer hash = Arrays.hashCode(instance.toDoubleArray());
            if (baseMap.containsKey(hash)) finalInstances.add(instance);
        }
        return finalInstances;
    }

    public static Instances findUnionBetweenInstances(Instances instances1, Instances instances2) {
        Instances allInstance = new Instances(instances1);
        allInstance.addAll(instances2);

        // Create a hashed mapped set of instances1 first
        Instances finalInstances = new Instances(instances1, instances1.size() + instances2.size());
        HashMap<Integer, Instance> baseMap = new HashMap<>(instances1.size());
        for (Instance instance : allInstance) {
            Integer hash = Arrays.hashCode(instance.toDoubleArray());
            if (!baseMap.containsKey(hash)) {
                baseMap.put(hash, instance);
                finalInstances.add(instance);
            }
        }
        return finalInstances;
    }
}
