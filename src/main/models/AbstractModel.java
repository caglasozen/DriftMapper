package main.models;

import main.models.prior.PriorModel;
import main.models.sampling.AbstractSampler;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Created by loongkuan on 1/04/16.
 */
public abstract class AbstractModel {
    // TODO: Change method to copy
    protected AbstractSampler sampler;
    public abstract AbstractModel copy();
    public abstract void reset();

    protected static Instances trimClass(Instances instances) {
        // Get attributes
        ArrayList<Attribute> attributes = new ArrayList<>();
        for (int i = 0; i < instances.numAttributes(); i++) {
            // if the current attribute is not a class attribute, add the attribute
            if (!(instances.classIndex() == i)) attributes.add(instances.attribute(i));
        }
        Instances instancesReturn = new Instances("Trimmed Data", attributes, instances.size());
        instancesReturn.setClassIndex(instancesReturn.numAttributes()-1);

        for (int i = 0; i < instances.size(); i++) {
            DenseInstance instance = new DenseInstance(instances.get(i));
            instance.deleteAttributeAt(instances.classIndex());
            instance.setDataset(instancesReturn);
            instancesReturn.add(instance);
        }
        return instancesReturn;
    }

    protected static Instances findIntersectionBetweenInstances(Instances instances1, Instances instances2) {
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

    protected static Instances findUnionBetweenInstances(Instances instances1, Instances instances2) {
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
