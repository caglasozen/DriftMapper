package main.models.frequent_sets;

import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 31/03/16.
 **/
public class Eclat {
    private HashMap<String[], Set<Integer>> database;
    public Eclat (Instances data) {
        database = convertData(data);
    }

    private HashMap<String[], Set<Integer>> convertData(Instances data) {
        HashMap<String[], Set<Integer>> hashMap = new HashMap<>();
        for (Instance inst : data) {
            for (int attIndex = 0; attIndex < inst.numAttributes(); attIndex++) {
                String item = Integer.toString(attIndex) + "_" + inst.attribute(attIndex).toString();
                if (!hashMap.containsKey(new String[]{item}))
                    hashMap.put(new String[]{item}, new HashSet<>(Arrays.hashCode(inst.toDoubleArray())));
                else hashMap.get(new String[]{item}).add(Arrays.hashCode(inst.toDoubleArray()));
            }
        }
        return hashMap;
    }

    public HashMap<String[], Set<Integer>> mineSet(){

        return new HashMap<>();
    }

    public HashMap<Set<String>, Set<Integer>> eclatCore(HashMap<Set<String>, Set<Integer>> database, int threshold){
        if (database.size() == 0) {
            return new HashMap<>();
        }
        else {
            HashMap<Set<String>, Set<Integer>> frequentSets = new HashMap<>();
            ArrayList<Set<String>> keys = new ArrayList<>(database.keySet());

            for (int i = 0; i < keys.size(); i++) {
                Set<String> baseKey = keys.get(i);
                frequentSets.putAll(database);
                HashMap<Set<String>, Set<Integer>> nextDatabase = new HashMap<>();

                for (int j = i + 1; j < keys.size(); j++) {
                    Set<String> currKey = keys.get(j);
                    // Intersection
                    Set<Integer> hashSet = database.get(baseKey);
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
