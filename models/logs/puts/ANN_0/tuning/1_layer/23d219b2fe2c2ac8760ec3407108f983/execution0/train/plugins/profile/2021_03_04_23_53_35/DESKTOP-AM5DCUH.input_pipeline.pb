	����r�2@����r�2@!����r�2@	���V�v?���V�v?!���V�v?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6����r�2@����1@1M2r���?AO��:7m�?IeV�p;t�?Y������P?*	43333sH@2U
Iterator::Model::ParallelMapV2%u��?!�"��>@)%u��?1�"��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!�|Bٹ�:@)��_vO�?1Ɩ���6@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�(��0�?!aV��C'9@)��ǘ���?1q�}�0@:Preprocessing2F
Iterator::Model/�$��?!A=S��xE@) �o_�y?1�0�Q�)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!��uǋ-!@)�J�4q?1��uǋ-!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip6�;Nё�?!�¬F�L@)��_�Le?1�έ�D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!L�����@)/n��b?1L�����@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�5.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���V�v?I��7Or�X@QY�-��L�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����1@����1@!����1@      ��!       "	M2r���?M2r���?!M2r���?*      ��!       2	O��:7m�?O��:7m�?!O��:7m�?:	eV�p;t�?eV�p;t�?!eV�p;t�?B      ��!       J	������P?������P?!������P?R      ��!       Z	������P?������P?!������P?b      ��!       JGPUY���V�v?b q��7Or�X@yY�-��L�?