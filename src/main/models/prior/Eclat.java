package main.models.prior;

import main.models.sampling.AbstractSampler;
import org.apache.commons.math3.analysis.function.Abs;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 31/03/16.
 **/
public class Eclat extends PriorModel{
    // Frequent Set -> Transaction ID
    private HashMap<Set<String>, Set<String>> frequentItemSet;
    private int threshold = 10;
    Instances dataSet;
    Instances allPossibleInstances;

    public Eclat (Instances dataSet) {
        this.dataSet = dataSet;
        reset();
    }

    public Eclat(AbstractSampler sampler) {
        this.dataSet = sampler.getDataSet();
        reset();
    }

    @Override
    public void reset() {
        HashMap<Set<String>, Set<String>> database;
        database = convertData(trimClass(this.dataSet));
        this.frequentItemSet = this.eclatCore(database, this.threshold);
        this.generateAllPossibleInstances();
    }

    @Override
    public Eclat copy() {
        return new Eclat(this.dataSet);
    }

    @Override
    public void setDataSet(Instances dataSet) {
        this.dataSet = dataSet;
        reset();
    }

    @Override
    public void setSampler(AbstractSampler sampler) {
        this.dataSet = sampler.getDataSet();
    }

    @Override
    //TODO: Process Partial Instances
    public double findPv(double[] vector) {
        Set<String> itemSet = this.convertInstance(vector);
        if (!this.frequentItemSet.containsKey(itemSet)) return 0.0f;
        return this.frequentItemSet.get(itemSet).size();
    }

    @Override
    public double findDistance(PriorModel model1, PriorModel model2, AbstractSampler sample) {
        if (sample.getNInstances() == 0) return 0.0f;
        double averageFrequency1 = 0;
        double averageFrequency2 = 0;
        sample.reset();
        for (int i = 0; i < sampler.getNInstances(); i++) {
            Instance inst = sample.nextInstance();
            averageFrequency1 += model1.findPv(inst.toDoubleArray());
            averageFrequency2 += model2.findPv(inst.toDoubleArray());
        }
        averageFrequency1 /= (double)sample.getNInstances();
        averageFrequency2 /= (double)sample.getNInstances();
        return Math.abs(averageFrequency1 - averageFrequency2);
    }

    private Set<String> convertInstance(double[] vector) {
        Set<String> itemSet = new HashSet<>();
        for (int i = 0; i < vector.length; i++) {
            if (!Double.isNaN(vector[i])) itemSet.add(Integer.toString(i) + "_" + Double.toString(vector[i]));
        }
        return itemSet;
    }

    private Instance convertSet(Set<String> set) {
        Instance instance = new DenseInstance(this.dataSet.numAttributes());
        for (String value : set) {
            String[] components = value.split("_");
            int attIndex = Integer.valueOf(components[0]);
            double attValue = Double.valueOf(components[1]);
            instance.setValue(attIndex, attValue);
        }
        return instance;
    }

    // TODO: Generate Partial Instances
    private void generateAllPossibleInstances() {
        this.allPossibleInstances = new Instances(this.dataSet, frequentItemSet.size());
        Set<Set<String>> partialSets = this.frequentItemSet.keySet();
        for (Set<String> partialSet : partialSets) {
            Instance partialInstance = convertSet(partialSet);
            this.allPossibleInstances.add(partialInstance);
        }
    }

    private HashMap<Set<String>, Set<String>> convertData(Instances data) {
        // Frequent Set -> Transaction IDs
        HashMap<Set<String>, Set<String>> hashMap = new HashMap<>();
        // Generate unfiltered vertical representation of database
        for (Instance inst : data) {
            for (int attIndex = 0; attIndex < inst.numAttributes(); attIndex++) {
                HashSet<String> itemSet = new HashSet<>();
                itemSet.add(Integer.toString(attIndex) + "_" + Double.toString(inst.toDoubleArray()[attIndex]));
                HashSet<String> tid = new HashSet<>();
                tid.add(inst.toString());
                if (!hashMap.containsKey(itemSet)) hashMap.put(itemSet, tid);
                else hashMap.get(itemSet).add(inst.toString());
            }
        }

        // Filter database based on number of occurence
        Set<Set<String>> keys = new HashSet<>(hashMap.keySet());
        for (Set<String> key : keys) {
            if (hashMap.get(key).size() < this.threshold) {
                hashMap.remove(key);
            }
        }
        return hashMap;
    }

    private HashMap<Set<String>, Set<String>> eclatCore(HashMap<Set<String>, Set<String>> database, int threshold){
        if (database.size() == 0) {
            return new HashMap<>();
        }
        else {
            ArrayList<Set<String>> keys = new ArrayList<>(database.keySet());

            HashMap<Set<String>, Set<String>> frequentSets = new HashMap<>(database);
            for (int i = 0; i < keys.size(); i++) {
                Set<String> baseKey = keys.get(i);
                HashMap<Set<String>, Set<String>> nextDatabase = new HashMap<>();

                for (int j = i + 1; j < keys.size(); j++) {
                    Set<String> currKey = keys.get(j);
                    // Intersection
                    Set<String> hashSet = new HashSet<>(database.get(baseKey));
                    hashSet.retainAll(database.get(currKey));
                    if (hashSet.size() >= threshold) {
                        Set<String> label = new HashSet<>(baseKey);
                        label.addAll(currKey);
                        nextDatabase.put(label, hashSet);
                    }
                }
                // Depth first recursion
                frequentSets.putAll(eclatCore(nextDatabase, threshold));
            }
            return frequentSets;
        }
    }
}
