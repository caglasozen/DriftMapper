package main.models.frequency;

import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Created by loongkuan on 26/11/16.
 **/

public class FrequencyMaps extends BaseFrequencyModel {
    private HashMap<int[], HashMap<Integer, int[]>> frequencyMaps;
    private int[] classFreq;
    private HashMap<int[], HashMap<Integer, Integer>> covariateFreq;

    public FrequencyMaps(Instances initialInstances, int nAttributesActive, int[] attributesAvailable) {
        this.attributesAvailable = attributesAvailable;
        this.nAttributesActive = nAttributesActive;
        this.exampleInst = initialInstances.firstInstance();
        this.classFreq = new int[this.exampleInst.numClasses()];
        this.covariateFreq = new HashMap<>();
        this.allInstances = initialInstances;

        // Get bases for hash
        int base = 1;
        this.hashBases = new int[initialInstances.numAttributes() - 1];
        for (int i = 0; i < initialInstances.numAttributes() - 1; i++) {
            this.hashBases[i] = base;
            base *= initialInstances.attribute(attributesAvailable[i]).numValues();
        }

        this.changeAttributeSubsetLength(nAttributesActive);
    }

    @Override
    public void changeAttributeSubsetLength(int length) {
        this.nAttributesActive = length;

        int nAttributeSubsets = FrequencyMaps.nCr(this.attributesAvailable.length, this.nAttributesActive);
        this.frequencyMaps = new HashMap<>();
        // Add a frequencyMap for each attribute subset
        for (int i = 0; i < nAttributeSubsets; i++) {
            this.frequencyMaps.put(getKthCombination(i, this.attributesAvailable, this.nAttributesActive), new HashMap<>());
        }
        // Add initialInstances to frequencyMaps
        for (Instance instance : this.allInstances) {
            this.addInstance(instance);
        }
    }

    @Override
    public void addInstance(Instance instance) {
        this.allInstances.add(instance);
        this.editInstance(instance, 1);
    }

    @Override
    public void removeInstance(Instance instance) {
        this.allInstances.remove(instance);
        this.editInstance(instance, -1);
    }

    @Override
    protected Set<Integer> getAllHashes(int[] attributeSubset) {
        return this.frequencyMaps.get(attributeSubset).keySet();
    }

    @Override
    protected int findFv(Instance instance, int[] attributesSubset) {
        int hash = BaseFrequencyModel.instanceToPartialHash(instance, attributesSubset, hashBases);
        return !hashSeen(hash, attributesSubset) ? 0 : this.frequencyMaps.get(attributesSubset).get(hash)[0];
    }

    @Override
    protected int findFy(int classIndex) {
        return this.classFreq[classIndex];
    }

    @Override
    protected int findFvy(Instance instance, int[] attributesSubset, int classIndex) {
        int hash = BaseFrequencyModel.instanceToPartialHash(instance, attributesSubset, hashBases);
        return !hashSeen(hash, attributesSubset) ? 0 : this.frequencyMaps.get(attributesSubset).get(hash)[1 + classIndex];
    }

    private boolean hashSeen(int hash, int[] attributeSubset) {
        return this.frequencyMaps.get(attributeSubset).containsKey(hash);
    }

    private void editInstance(Instance instance, int amount) {
        this.totalFrequency += amount;
        this.classFreq[(int)instance.classValue()] += amount;
        int nAttributeSubsets = nCr(this.attributesAvailable.length, this.nAttributesActive);
        for (int j = 0; j < nAttributeSubsets; j++) {
            int[] activeAttributes = getKthCombination(j, attributesAvailable, nAttributesActive);
            int partialHash = FrequencyMaps.instanceToPartialHash(instance, activeAttributes, hashBases);
            if (!this.frequencyMaps.get(activeAttributes).containsKey(partialHash)) {
                this.frequencyMaps.get(activeAttributes).put(partialHash, new int[1 + this.exampleInst.numClasses()]);
            }
            int[] prevFreq = this.frequencyMaps.get(activeAttributes).get(partialHash);
            prevFreq[0] += amount;
            prevFreq[(int)instance.classValue()] += amount;
            this.frequencyMaps.get(activeAttributes).replace(partialHash, prevFreq);
        }
    }

}
