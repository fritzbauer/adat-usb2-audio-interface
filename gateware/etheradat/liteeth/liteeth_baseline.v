/* Machine-generated using Migen */
module liteeth_baseline(
	output rmii_tx_en,
	output [1:0] rmii_txd,
	input rmii_crs_dv,
	input [1:0] rmii_rxd,
	output dbg_crs_dv,
	output dbg_rx_udp,
	output dbg_rx_mac,
	output dbg_tx_mac,
	output dbg_pre_err,
	output dbg_crc_err,
	output dbg_arp_rx,
	output dbg_arp_req,
	output dbg_arp_rep,
	output dbg_mac_in,
	output dbg_mac_out,
	output dbg_led_shim_last,
	output dbg_led_cdc_last,
	output dbg_led_mac_last,
	output dbg_led_arp_req,
	output dbg_led_arp_rep,
	output dbg_led_tx_last,
	output dbg_led_rx_err,
	input sys_clk,
	input sys_rst,
	input eth_tx_clk,
	input eth_tx_rst,
	input eth_rx_clk,
	input eth_rx_rst,
	output uart_txd
);

wire phy_sink_valid;
wire phy_sink_ready;
wire phy_sink_first;
wire phy_sink_last;
wire [7:0] phy_sink_payload_data;
wire phy_sink_payload_be;
wire phy_sink_payload_error;
wire phy_source_valid;
wire phy_source_ready;
wire phy_source_first;
wire phy_source_last;
wire [7:0] phy_source_payload_data;
wire phy_source_payload_be;
wire phy_source_payload_error;
wire phy_rmiitxshim_sink_valid;
wire phy_rmiitxshim_sink_ready;
wire phy_rmiitxshim_sink_first;
wire phy_rmiitxshim_sink_last;
wire [7:0] phy_rmiitxshim_sink_payload_data;
wire phy_rmiitxshim_sink_payload_be;
wire phy_rmiitxshim_sink_payload_error;
reg [7:0] phy_rmiitxshim_shift = 8'd0;
reg [2:0] phy_rmiitxshim_count = 3'd0;
wire phy_rmiirxshim_source_valid;
wire phy_rmiirxshim_source_ready;
reg phy_rmiirxshim_source_first = 1'd0;
wire phy_rmiirxshim_source_last;
wire [7:0] phy_rmiirxshim_source_payload_data;
wire phy_rmiirxshim_source_payload_be;
reg phy_rmiirxshim_source_payload_error = 1'd0;
wire phy_rmiirxshim_crs_dv;
wire [1:0] phy_rmiirxshim_rx_data;
wire phy_rmiirxshim_crs_first;
wire phy_rmiirxshim_crs_last;
reg phy_rmiirxshim_crs_run = 1'd0;
reg phy_rmiirxshim_crs_dv_d = 1'd0;
reg [1:0] phy_rmiirxshim_d0 = 2'd0;
reg [1:0] phy_rmiirxshim_d1 = 2'd0;
reg phy_rmiirxshim_e0 = 1'd0;
reg phy_rmiirxshim_e1 = 1'd0;
wire phy_rmiirxshim_cap;
reg [1:0] phy_rmiirxshim_mode = 2'd0;
reg phy_rmiirxshim_f0_valid = 1'd0;
reg [7:0] phy_rmiirxshim_f0_data = 8'd0;
reg phy_rmiirxshim_f0_last = 1'd0;
reg phy_rmiirxshim_f1_valid = 1'd0;
reg [7:0] phy_rmiirxshim_f1_data = 8'd0;
reg phy_rmiirxshim_f1_last = 1'd0;
reg [7:0] phy_rmiirxshim_sreg = 8'd0;
wire phy_rmiirxshim_sfd_hit;
reg [2:0] phy_rmiirxshim_pre_cnt = 3'd0;
wire [7:0] phy_rmiirxshim_pre_byte;
reg [7:0] phy_rmiirxshim_shift = 8'd0;
reg [1:0] phy_rmiirxshim_bits = 2'd0;
wire phy_rmiirxshim_consume;
wire [7:0] phy_rmiirxshim_new_byte;
wire phy_rmiirxshim_pushing;
wire mac_core_sink_valid;
wire mac_core_sink_ready;
wire mac_core_sink_first;
wire mac_core_sink_last;
wire [7:0] mac_core_sink_payload_data;
wire mac_core_sink_payload_be;
wire mac_core_sink_payload_error;
wire mac_core_source_valid;
wire mac_core_source_ready;
wire mac_core_source_first;
wire mac_core_source_last;
wire [7:0] mac_core_source_payload_data;
wire mac_core_source_payload_be;
wire mac_core_source_payload_error;
wire mac_core_cdc_rst;
wire mac_core_tx_padding_sink_valid;
reg mac_core_tx_padding_sink_ready;
wire mac_core_tx_padding_sink_first;
wire mac_core_tx_padding_sink_last;
wire [7:0] mac_core_tx_padding_sink_payload_data;
wire mac_core_tx_padding_sink_payload_be;
wire mac_core_tx_padding_sink_payload_error;
reg mac_core_tx_padding_source_valid;
wire mac_core_tx_padding_source_ready;
reg mac_core_tx_padding_source_first;
reg mac_core_tx_padding_source_last;
reg [7:0] mac_core_tx_padding_source_payload_data;
reg mac_core_tx_padding_source_payload_be;
reg mac_core_tx_padding_source_payload_error;
reg [15:0] mac_core_tx_padding_counter = 16'd0;
wire mac_core_tx_padding_counter_done;
wire mac_core_tx_crc_sink_valid;
reg mac_core_tx_crc_sink_ready;
wire mac_core_tx_crc_sink_first;
wire mac_core_tx_crc_sink_last;
wire [7:0] mac_core_tx_crc_sink_payload_data;
wire mac_core_tx_crc_sink_payload_be;
wire mac_core_tx_crc_sink_payload_error;
reg mac_core_tx_crc_source_valid;
wire mac_core_tx_crc_source_ready;
reg mac_core_tx_crc_source_first;
reg mac_core_tx_crc_source_last;
reg [7:0] mac_core_tx_crc_source_payload_data;
reg mac_core_tx_crc_source_payload_be;
reg mac_core_tx_crc_source_payload_error;
reg [31:0] mac_core_tx_crc_crc_packet = 32'd0;
reg mac_core_tx_crc_be0 = 1'd0;
wire [7:0] mac_core_tx_crc_data0;
wire mac_core_tx_crc_be1;
reg [31:0] mac_core_tx_crc_value;
reg mac_core_tx_crc_error;
wire [7:0] mac_core_tx_crc_data1;
wire [31:0] mac_core_tx_crc_crc_prev;
reg [31:0] mac_core_tx_crc_crc_next;
reg [31:0] mac_core_tx_crc_reg = 32'd4294967295;
reg mac_core_tx_crc_ce;
reg mac_core_tx_crc_reset;
reg [1:0] mac_core_tx_crc_cnt = 2'd3;
wire mac_core_tx_crc_cnt_done;
reg mac_core_tx_crc_is_ongoing0;
reg mac_core_tx_crc_is_ongoing1;
wire mac_core_tx_crc_sink_sink_valid;
wire mac_core_tx_crc_sink_sink_ready;
wire mac_core_tx_crc_sink_sink_first;
wire mac_core_tx_crc_sink_sink_last;
wire [7:0] mac_core_tx_crc_sink_sink_payload_data;
wire mac_core_tx_crc_sink_sink_payload_be;
wire mac_core_tx_crc_sink_sink_payload_error;
wire mac_core_tx_crc_source_source_valid;
wire mac_core_tx_crc_source_source_ready;
wire mac_core_tx_crc_source_source_first;
wire mac_core_tx_crc_source_source_last;
wire [7:0] mac_core_tx_crc_source_source_payload_data;
wire mac_core_tx_crc_source_source_payload_be;
wire mac_core_tx_crc_source_source_payload_error;
wire mac_core_tx_crc_pipe_valid_sink_valid;
wire mac_core_tx_crc_pipe_valid_sink_ready;
wire mac_core_tx_crc_pipe_valid_sink_first;
wire mac_core_tx_crc_pipe_valid_sink_last;
wire [7:0] mac_core_tx_crc_pipe_valid_sink_payload_data;
wire mac_core_tx_crc_pipe_valid_sink_payload_be;
wire mac_core_tx_crc_pipe_valid_sink_payload_error;
reg mac_core_tx_crc_pipe_valid_source_valid = 1'd0;
wire mac_core_tx_crc_pipe_valid_source_ready;
reg mac_core_tx_crc_pipe_valid_source_first = 1'd0;
reg mac_core_tx_crc_pipe_valid_source_last = 1'd0;
reg [7:0] mac_core_tx_crc_pipe_valid_source_payload_data = 8'd0;
reg mac_core_tx_crc_pipe_valid_source_payload_be = 1'd0;
reg mac_core_tx_crc_pipe_valid_source_payload_error = 1'd0;
wire mac_core_tx_preamble_sink_valid;
reg mac_core_tx_preamble_sink_ready;
wire mac_core_tx_preamble_sink_first;
wire mac_core_tx_preamble_sink_last;
wire [7:0] mac_core_tx_preamble_sink_payload_data;
wire mac_core_tx_preamble_sink_payload_be;
wire mac_core_tx_preamble_sink_payload_error;
reg mac_core_tx_preamble_source_valid;
wire mac_core_tx_preamble_source_ready;
reg mac_core_tx_preamble_source_first;
reg mac_core_tx_preamble_source_last;
reg [7:0] mac_core_tx_preamble_source_payload_data;
reg mac_core_tx_preamble_source_payload_be;
reg mac_core_tx_preamble_source_payload_error;
reg [63:0] mac_core_tx_preamble_preamble = 64'd15372286728091293013;
reg [2:0] mac_core_tx_preamble_count = 3'd0;
wire mac_core_txdatapath_sink_valid;
wire mac_core_txdatapath_sink_ready;
wire mac_core_txdatapath_sink_first;
wire mac_core_txdatapath_sink_last;
wire [7:0] mac_core_txdatapath_sink_payload_data;
wire mac_core_txdatapath_sink_payload_be;
wire mac_core_txdatapath_sink_payload_error;
wire mac_core_txdatapath_source_valid;
wire mac_core_txdatapath_source_ready;
wire mac_core_txdatapath_source_first;
wire mac_core_txdatapath_source_last;
wire [7:0] mac_core_txdatapath_source_payload_data;
wire mac_core_txdatapath_source_payload_be;
wire mac_core_txdatapath_source_payload_error;
reg mac_core_txdatapath_ext_rst_to = 1'd0;
wire mac_core_txdatapath_sink_sink_valid;
wire mac_core_txdatapath_sink_sink_ready;
wire mac_core_txdatapath_sink_sink_first;
wire mac_core_txdatapath_sink_sink_last;
wire [7:0] mac_core_txdatapath_sink_sink_payload_data;
wire mac_core_txdatapath_sink_sink_payload_be;
wire mac_core_txdatapath_sink_sink_payload_error;
wire mac_core_txdatapath_source_source_valid;
wire mac_core_txdatapath_source_source_ready;
wire mac_core_txdatapath_source_source_first;
wire mac_core_txdatapath_source_source_last;
wire [7:0] mac_core_txdatapath_source_source_payload_data;
wire mac_core_txdatapath_source_source_payload_be;
wire mac_core_txdatapath_source_source_payload_error;
wire mac_core_txdatapath_pipe_valid_sink_valid;
wire mac_core_txdatapath_pipe_valid_sink_ready;
wire mac_core_txdatapath_pipe_valid_sink_first;
wire mac_core_txdatapath_pipe_valid_sink_last;
wire [7:0] mac_core_txdatapath_pipe_valid_sink_payload_data;
wire mac_core_txdatapath_pipe_valid_sink_payload_be;
wire mac_core_txdatapath_pipe_valid_sink_payload_error;
reg mac_core_txdatapath_pipe_valid_source_valid = 1'd0;
wire mac_core_txdatapath_pipe_valid_source_ready;
reg mac_core_txdatapath_pipe_valid_source_first = 1'd0;
reg mac_core_txdatapath_pipe_valid_source_last = 1'd0;
reg [7:0] mac_core_txdatapath_pipe_valid_source_payload_data = 8'd0;
reg mac_core_txdatapath_pipe_valid_source_payload_be = 1'd0;
reg mac_core_txdatapath_pipe_valid_source_payload_error = 1'd0;
wire mac_core_tx_gap_sink_valid;
reg mac_core_tx_gap_sink_ready;
wire mac_core_tx_gap_sink_first;
wire mac_core_tx_gap_sink_last;
wire [7:0] mac_core_tx_gap_sink_payload_data;
wire mac_core_tx_gap_sink_payload_be;
wire mac_core_tx_gap_sink_payload_error;
reg mac_core_tx_gap_source_valid;
wire mac_core_tx_gap_source_ready;
reg mac_core_tx_gap_source_first;
reg mac_core_tx_gap_source_last;
reg [7:0] mac_core_tx_gap_source_payload_data;
reg mac_core_tx_gap_source_payload_be;
reg mac_core_tx_gap_source_payload_error;
reg [3:0] mac_core_tx_gap_counter = 4'd0;
wire mac_core_tx_gap_reset;
reg [31:0] mac_core_preamble_errors_status = 32'd0;
reg [31:0] mac_core_crc_errors_status = 32'd0;
wire mac_core_rxdatapath_sink_valid;
wire mac_core_rxdatapath_sink_ready;
wire mac_core_rxdatapath_sink_first;
wire mac_core_rxdatapath_sink_last;
wire [7:0] mac_core_rxdatapath_sink_payload_data;
wire mac_core_rxdatapath_sink_payload_be;
wire mac_core_rxdatapath_sink_payload_error;
wire mac_core_rxdatapath_source_valid;
wire mac_core_rxdatapath_source_ready;
wire mac_core_rxdatapath_source_first;
wire mac_core_rxdatapath_source_last;
wire [7:0] mac_core_rxdatapath_source_payload_data;
wire mac_core_rxdatapath_source_payload_be;
wire mac_core_rxdatapath_source_payload_error;
wire mac_core_rxdatapath_sink_sink_valid;
wire mac_core_rxdatapath_sink_sink_ready;
wire mac_core_rxdatapath_sink_sink_first;
wire mac_core_rxdatapath_sink_sink_last;
wire [7:0] mac_core_rxdatapath_sink_sink_payload_data;
wire mac_core_rxdatapath_sink_sink_payload_be;
wire mac_core_rxdatapath_sink_sink_payload_error;
wire mac_core_rxdatapath_source_source_valid;
wire mac_core_rxdatapath_source_source_ready;
wire mac_core_rxdatapath_source_source_first;
wire mac_core_rxdatapath_source_source_last;
wire [7:0] mac_core_rxdatapath_source_source_payload_data;
wire mac_core_rxdatapath_source_source_payload_be;
wire mac_core_rxdatapath_source_source_payload_error;
wire mac_core_rxdatapath_pipe_valid_sink_valid;
wire mac_core_rxdatapath_pipe_valid_sink_ready;
wire mac_core_rxdatapath_pipe_valid_sink_first;
wire mac_core_rxdatapath_pipe_valid_sink_last;
wire [7:0] mac_core_rxdatapath_pipe_valid_sink_payload_data;
wire mac_core_rxdatapath_pipe_valid_sink_payload_be;
wire mac_core_rxdatapath_pipe_valid_sink_payload_error;
reg mac_core_rxdatapath_pipe_valid_source_valid = 1'd0;
wire mac_core_rxdatapath_pipe_valid_source_ready;
reg mac_core_rxdatapath_pipe_valid_source_first = 1'd0;
reg mac_core_rxdatapath_pipe_valid_source_last = 1'd0;
reg [7:0] mac_core_rxdatapath_pipe_valid_source_payload_data = 8'd0;
reg mac_core_rxdatapath_pipe_valid_source_payload_be = 1'd0;
reg mac_core_rxdatapath_pipe_valid_source_payload_error = 1'd0;
wire mac_core_rx_preamble_sink_valid;
reg mac_core_rx_preamble_sink_ready;
wire mac_core_rx_preamble_sink_first;
wire mac_core_rx_preamble_sink_last;
wire [7:0] mac_core_rx_preamble_sink_payload_data;
wire mac_core_rx_preamble_sink_payload_be;
wire mac_core_rx_preamble_sink_payload_error;
reg mac_core_rx_preamble_source_valid;
wire mac_core_rx_preamble_source_ready;
reg mac_core_rx_preamble_source_first;
reg mac_core_rx_preamble_source_last;
wire [7:0] mac_core_rx_preamble_source_payload_data;
wire mac_core_rx_preamble_source_payload_be;
reg mac_core_rx_preamble_source_payload_error;
reg mac_core_rx_preamble_error;
reg [63:0] mac_core_rx_preamble_preamble = 64'd15372286728091293013;
wire mac_core_pulsesynchronizer0_i;
wire mac_core_pulsesynchronizer0_o;
reg mac_core_pulsesynchronizer0_toggle_i = 1'd0;
wire mac_core_pulsesynchronizer0_toggle_o;
reg mac_core_pulsesynchronizer0_toggle_o_r = 1'd0;
wire mac_core_liteethmaccrc32checker_sink_sink_valid;
reg mac_core_liteethmaccrc32checker_sink_sink_ready;
wire mac_core_liteethmaccrc32checker_sink_sink_first;
wire mac_core_liteethmaccrc32checker_sink_sink_last;
wire [7:0] mac_core_liteethmaccrc32checker_sink_sink_payload_data;
wire mac_core_liteethmaccrc32checker_sink_sink_payload_be;
wire mac_core_liteethmaccrc32checker_sink_sink_payload_error;
reg mac_core_liteethmaccrc32checker_source_source_valid;
wire mac_core_liteethmaccrc32checker_source_source_ready;
wire mac_core_liteethmaccrc32checker_source_source_first;
reg mac_core_liteethmaccrc32checker_source_source_last;
wire [7:0] mac_core_liteethmaccrc32checker_source_source_payload_data;
reg mac_core_liteethmaccrc32checker_source_source_payload_be;
reg mac_core_liteethmaccrc32checker_source_source_payload_error;
reg mac_core_liteethmaccrc32checker_error;
wire [7:0] mac_core_liteethmaccrc32checker_crc_data0;
wire mac_core_liteethmaccrc32checker_crc_be;
reg mac_core_liteethmaccrc32checker_crc_error;
reg [7:0] mac_core_liteethmaccrc32checker_crc_data1;
wire [31:0] mac_core_liteethmaccrc32checker_crc_crc_prev;
reg [31:0] mac_core_liteethmaccrc32checker_crc_crc_next;
reg [31:0] mac_core_liteethmaccrc32checker_crc_reg = 32'd4294967295;
reg mac_core_liteethmaccrc32checker_crc_ce;
reg mac_core_liteethmaccrc32checker_crc_reset;
reg mac_core_liteethmaccrc32checker_syncfifo_sink_valid;
wire mac_core_liteethmaccrc32checker_syncfifo_sink_ready;
wire mac_core_liteethmaccrc32checker_syncfifo_sink_first;
wire mac_core_liteethmaccrc32checker_syncfifo_sink_last;
wire [7:0] mac_core_liteethmaccrc32checker_syncfifo_sink_payload_data;
wire mac_core_liteethmaccrc32checker_syncfifo_sink_payload_be;
wire mac_core_liteethmaccrc32checker_syncfifo_sink_payload_error;
wire mac_core_liteethmaccrc32checker_syncfifo_source_valid;
reg mac_core_liteethmaccrc32checker_syncfifo_source_ready;
wire mac_core_liteethmaccrc32checker_syncfifo_source_first;
wire mac_core_liteethmaccrc32checker_syncfifo_source_last;
wire [7:0] mac_core_liteethmaccrc32checker_syncfifo_source_payload_data;
wire mac_core_liteethmaccrc32checker_syncfifo_source_payload_be;
wire mac_core_liteethmaccrc32checker_syncfifo_source_payload_error;
wire mac_core_liteethmaccrc32checker_syncfifo_syncfifo_we;
wire mac_core_liteethmaccrc32checker_syncfifo_syncfifo_writable;
wire mac_core_liteethmaccrc32checker_syncfifo_syncfifo_re;
wire mac_core_liteethmaccrc32checker_syncfifo_syncfifo_readable;
wire [11:0] mac_core_liteethmaccrc32checker_syncfifo_syncfifo_din;
wire [11:0] mac_core_liteethmaccrc32checker_syncfifo_syncfifo_dout;
reg [2:0] mac_core_liteethmaccrc32checker_syncfifo_level = 3'd0;
reg mac_core_liteethmaccrc32checker_syncfifo_replace = 1'd0;
reg [2:0] mac_core_liteethmaccrc32checker_syncfifo_produce = 3'd0;
reg [2:0] mac_core_liteethmaccrc32checker_syncfifo_consume = 3'd0;
reg [2:0] mac_core_liteethmaccrc32checker_syncfifo_wrport_adr;
wire mac_core_liteethmaccrc32checker_syncfifo_wrport_we;
wire [11:0] mac_core_liteethmaccrc32checker_syncfifo_wrport_dat_w;
wire mac_core_liteethmaccrc32checker_syncfifo_do_read;
wire [2:0] mac_core_liteethmaccrc32checker_syncfifo_rdport_adr;
wire [11:0] mac_core_liteethmaccrc32checker_syncfifo_rdport_dat_r;
wire [7:0] mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_data;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_be;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_error;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_in_first;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_in_last;
wire [7:0] mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_data;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_be;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_error;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_out_first;
wire mac_core_liteethmaccrc32checker_syncfifo_fifo_out_last;
reg mac_core_liteethmaccrc32checker_fifo_reset;
wire mac_core_liteethmaccrc32checker_fifo_in;
wire mac_core_liteethmaccrc32checker_fifo_out;
wire mac_core_liteethmaccrc32checker_fifo_full;
reg mac_core_liteethmaccrc32checker_is_ongoing0;
reg mac_core_liteethmaccrc32checker_is_ongoing1;
reg mac_core_liteethmaccrc32checker_be = 1'd0;
reg mac_core_liteethmaccrc32checker_last_error = 1'd0;
wire mac_core_liteethmaccrc32checker_last_word_error;
reg mac_core_liteethmaccrc32checker_packet_error = 1'd0;
reg mac_core_liteethmaccrc32checker_is_ongoing2;
wire mac_core_bufferizeendpoints_sink_sink_valid;
wire mac_core_bufferizeendpoints_sink_sink_ready;
wire mac_core_bufferizeendpoints_sink_sink_first;
wire mac_core_bufferizeendpoints_sink_sink_last;
wire [7:0] mac_core_bufferizeendpoints_sink_sink_payload_data;
wire mac_core_bufferizeendpoints_sink_sink_payload_be;
wire mac_core_bufferizeendpoints_sink_sink_payload_error;
wire mac_core_bufferizeendpoints_source_source_valid;
wire mac_core_bufferizeendpoints_source_source_ready;
wire mac_core_bufferizeendpoints_source_source_first;
wire mac_core_bufferizeendpoints_source_source_last;
wire [7:0] mac_core_bufferizeendpoints_source_source_payload_data;
wire mac_core_bufferizeendpoints_source_source_payload_be;
wire mac_core_bufferizeendpoints_source_source_payload_error;
wire mac_core_bufferizeendpoints_pipe_valid_sink_valid;
wire mac_core_bufferizeendpoints_pipe_valid_sink_ready;
wire mac_core_bufferizeendpoints_pipe_valid_sink_first;
wire mac_core_bufferizeendpoints_pipe_valid_sink_last;
wire [7:0] mac_core_bufferizeendpoints_pipe_valid_sink_payload_data;
wire mac_core_bufferizeendpoints_pipe_valid_sink_payload_be;
wire mac_core_bufferizeendpoints_pipe_valid_sink_payload_error;
reg mac_core_bufferizeendpoints_pipe_valid_source_valid = 1'd0;
wire mac_core_bufferizeendpoints_pipe_valid_source_ready;
reg mac_core_bufferizeendpoints_pipe_valid_source_first = 1'd0;
reg mac_core_bufferizeendpoints_pipe_valid_source_last = 1'd0;
reg [7:0] mac_core_bufferizeendpoints_pipe_valid_source_payload_data = 8'd0;
reg mac_core_bufferizeendpoints_pipe_valid_source_payload_be = 1'd0;
reg mac_core_bufferizeendpoints_pipe_valid_source_payload_error = 1'd0;
wire mac_core_pulsesynchronizer1_i;
wire mac_core_pulsesynchronizer1_o;
reg mac_core_pulsesynchronizer1_toggle_i = 1'd0;
wire mac_core_pulsesynchronizer1_toggle_o;
reg mac_core_pulsesynchronizer1_toggle_o_r = 1'd0;
wire mac_core_rx_padding_sink_valid;
wire mac_core_rx_padding_sink_ready;
wire mac_core_rx_padding_sink_first;
wire mac_core_rx_padding_sink_last;
wire [7:0] mac_core_rx_padding_sink_payload_data;
wire mac_core_rx_padding_sink_payload_be;
wire mac_core_rx_padding_sink_payload_error;
wire mac_core_rx_padding_source_valid;
wire mac_core_rx_padding_source_ready;
wire mac_core_rx_padding_source_first;
wire mac_core_rx_padding_source_last;
wire [7:0] mac_core_rx_padding_source_payload_data;
wire mac_core_rx_padding_source_payload_be;
reg mac_core_rx_padding_source_payload_error;
reg [10:0] mac_core_rx_padding_length = 11'd0;
wire [3:0] mac_core_rx_padding_length_inc;
reg mac_crossbar_source_valid;
wire mac_crossbar_source_ready;
reg mac_crossbar_source_first;
reg mac_crossbar_source_last;
reg [15:0] mac_crossbar_source_payload_ethernet_type;
reg [47:0] mac_crossbar_source_payload_sender_mac;
reg [47:0] mac_crossbar_source_payload_target_mac;
reg [7:0] mac_crossbar_source_payload_data;
reg mac_crossbar_source_payload_be;
reg mac_crossbar_source_payload_error;
wire mac_crossbar_sink_valid;
reg mac_crossbar_sink_ready;
wire mac_crossbar_sink_first;
wire mac_crossbar_sink_last;
wire [15:0] mac_crossbar_sink_payload_ethernet_type;
wire [47:0] mac_crossbar_sink_payload_sender_mac;
wire [47:0] mac_crossbar_sink_payload_target_mac;
wire [7:0] mac_crossbar_sink_payload_data;
wire mac_crossbar_sink_payload_be;
wire mac_crossbar_sink_payload_error;
wire mac_packetizer_sink_valid;
reg mac_packetizer_sink_ready;
wire mac_packetizer_sink_first;
wire mac_packetizer_sink_last;
wire [15:0] mac_packetizer_sink_payload_ethernet_type;
wire [47:0] mac_packetizer_sink_payload_sender_mac;
wire [47:0] mac_packetizer_sink_payload_target_mac;
wire [7:0] mac_packetizer_sink_payload_data;
wire mac_packetizer_sink_payload_be;
wire mac_packetizer_sink_payload_error;
reg mac_packetizer_source_valid;
wire mac_packetizer_source_ready;
reg mac_packetizer_source_first;
reg mac_packetizer_source_last;
reg [7:0] mac_packetizer_source_payload_data;
wire mac_packetizer_source_payload_be;
reg mac_packetizer_source_payload_error;
reg [111:0] mac_packetizer_header;
reg mac_packetizer_source_be;
reg [3:0] mac_packetizer_count = 4'd0;
reg mac_packetizer_is_ongoing;
wire mac_depacketizer_sink_valid;
reg mac_depacketizer_sink_ready;
wire mac_depacketizer_sink_first;
wire mac_depacketizer_sink_last;
wire [7:0] mac_depacketizer_sink_payload_data;
wire mac_depacketizer_sink_payload_be;
wire mac_depacketizer_sink_payload_error;
reg mac_depacketizer_source_valid;
wire mac_depacketizer_source_ready;
reg mac_depacketizer_source_first;
reg mac_depacketizer_source_last;
wire [15:0] mac_depacketizer_source_payload_ethernet_type;
wire [47:0] mac_depacketizer_source_payload_sender_mac;
wire [47:0] mac_depacketizer_source_payload_target_mac;
reg [7:0] mac_depacketizer_source_payload_data;
reg mac_depacketizer_source_payload_be;
wire mac_depacketizer_source_payload_error;
wire [111:0] mac_depacketizer_header;
reg [111:0] mac_depacketizer_sr = 112'd0;
reg [3:0] mac_depacketizer_count = 4'd0;
reg mac_depacketizer_skip_cnt = 1'd0;
reg mac_depacketizer_data_copy_first = 1'd1;
wire mac_depacketizer_sr_shift;
reg mac_depacketizer_is_ongoing0;
reg mac_depacketizer_is_ongoing1;
reg mac_depacketizer_send_dummy = 1'd0;
wire arp_tx_sink_sink_valid;
reg arp_tx_sink_sink_ready;
wire arp_tx_sink_sink_first;
wire arp_tx_sink_sink_last;
wire arp_tx_sink_sink_payload_reply;
wire arp_tx_sink_sink_payload_request;
wire [31:0] arp_tx_sink_sink_payload_ip_address;
wire [47:0] arp_tx_sink_sink_payload_mac_address;
reg arp_tx_source_source_valid;
wire arp_tx_source_source_ready;
wire arp_tx_source_source_first;
wire arp_tx_source_source_last;
reg [15:0] arp_tx_source_source_payload_ethernet_type;
reg [47:0] arp_tx_source_source_payload_sender_mac;
reg [47:0] arp_tx_source_source_payload_target_mac;
wire [7:0] arp_tx_source_source_payload_data;
wire arp_tx_source_source_payload_be;
wire arp_tx_source_source_payload_error;
reg [4:0] arp_tx_counter = 5'd0;
reg arp_tx_packetizer_sink_valid;
reg arp_tx_packetizer_sink_ready;
wire arp_tx_packetizer_sink_last;
reg [7:0] arp_tx_packetizer_sink_payload_data = 8'd0;
wire arp_tx_packetizer_sink_payload_be;
reg arp_tx_packetizer_sink_payload_error = 1'd0;
wire [7:0] arp_tx_packetizer_sink_param_hwsize;
wire [15:0] arp_tx_packetizer_sink_param_hwtype;
reg [15:0] arp_tx_packetizer_sink_param_opcode;
wire [15:0] arp_tx_packetizer_sink_param_proto;
wire [7:0] arp_tx_packetizer_sink_param_protosize;
wire [31:0] arp_tx_packetizer_sink_param_sender_ip;
wire [47:0] arp_tx_packetizer_sink_param_sender_mac;
wire [31:0] arp_tx_packetizer_sink_param_target_ip;
reg [47:0] arp_tx_packetizer_sink_param_target_mac;
reg arp_tx_packetizer_source_valid;
reg arp_tx_packetizer_source_ready;
reg arp_tx_packetizer_source_first;
reg arp_tx_packetizer_source_last;
reg [15:0] arp_tx_packetizer_source_payload_ethernet_type = 16'd0;
reg [47:0] arp_tx_packetizer_source_payload_sender_mac = 48'd0;
reg [47:0] arp_tx_packetizer_source_payload_target_mac = 48'd0;
reg [7:0] arp_tx_packetizer_source_payload_data;
wire arp_tx_packetizer_source_payload_be;
reg arp_tx_packetizer_source_payload_error;
reg [223:0] arp_tx_packetizer_header;
reg arp_tx_packetizer_source_be;
reg [4:0] arp_tx_packetizer_count = 5'd0;
reg arp_tx_packetizer_is_ongoing;
wire arp_rx_sink_sink_valid;
wire arp_rx_sink_sink_ready;
wire arp_rx_sink_sink_first;
wire arp_rx_sink_sink_last;
wire [15:0] arp_rx_sink_sink_payload_ethernet_type;
wire [47:0] arp_rx_sink_sink_payload_sender_mac;
wire [47:0] arp_rx_sink_sink_payload_target_mac;
wire [7:0] arp_rx_sink_sink_payload_data;
wire arp_rx_sink_sink_payload_be;
wire arp_rx_sink_sink_payload_error;
reg arp_rx_source_source_valid;
wire arp_rx_source_source_ready;
reg arp_rx_source_source_first = 1'd0;
reg arp_rx_source_source_last = 1'd0;
reg arp_rx_source_source_payload_reply;
reg arp_rx_source_source_payload_request;
wire [31:0] arp_rx_source_source_payload_ip_address;
wire [47:0] arp_rx_source_source_payload_mac_address;
wire arp_rx_depacketizer_sink_valid;
reg arp_rx_depacketizer_sink_ready;
wire arp_rx_depacketizer_sink_first;
wire arp_rx_depacketizer_sink_last;
wire [15:0] arp_rx_depacketizer_sink_payload_ethernet_type;
wire [47:0] arp_rx_depacketizer_sink_payload_sender_mac;
wire [47:0] arp_rx_depacketizer_sink_payload_target_mac;
wire [7:0] arp_rx_depacketizer_sink_payload_data;
wire arp_rx_depacketizer_sink_payload_be;
wire arp_rx_depacketizer_sink_payload_error;
reg arp_rx_depacketizer_source_valid;
reg arp_rx_depacketizer_source_ready;
reg arp_rx_depacketizer_source_first;
reg arp_rx_depacketizer_source_last;
reg [7:0] arp_rx_depacketizer_source_payload_data;
reg arp_rx_depacketizer_source_payload_be;
wire arp_rx_depacketizer_source_payload_error;
wire [7:0] arp_rx_depacketizer_source_param_hwsize;
wire [15:0] arp_rx_depacketizer_source_param_hwtype;
wire [15:0] arp_rx_depacketizer_source_param_opcode;
wire [15:0] arp_rx_depacketizer_source_param_proto;
wire [7:0] arp_rx_depacketizer_source_param_protosize;
wire [31:0] arp_rx_depacketizer_source_param_sender_ip;
wire [47:0] arp_rx_depacketizer_source_param_sender_mac;
wire [31:0] arp_rx_depacketizer_source_param_target_ip;
wire [47:0] arp_rx_depacketizer_source_param_target_mac;
wire [223:0] arp_rx_depacketizer_header;
reg [223:0] arp_rx_depacketizer_sr = 224'd0;
reg [4:0] arp_rx_depacketizer_count = 5'd0;
reg arp_rx_depacketizer_skip_cnt = 1'd0;
reg arp_rx_depacketizer_data_copy_first = 1'd1;
wire arp_rx_depacketizer_sr_shift;
reg arp_rx_depacketizer_is_ongoing0;
reg arp_rx_depacketizer_is_ongoing1;
reg arp_rx_depacketizer_send_dummy = 1'd0;
reg arp_rx_valid = 1'd0;
reg arp_rx_reply;
reg arp_rx_request;
wire arp_table_sink_valid;
reg arp_table_sink_ready = 1'd0;
wire arp_table_sink_first;
wire arp_table_sink_last;
wire arp_table_sink_payload_reply;
wire arp_table_sink_payload_request;
wire [31:0] arp_table_sink_payload_ip_address;
wire [47:0] arp_table_sink_payload_mac_address;
reg arp_table_source_valid;
wire arp_table_source_ready;
reg arp_table_source_first = 1'd0;
reg arp_table_source_last = 1'd0;
reg arp_table_source_payload_reply;
reg arp_table_source_payload_request;
reg [31:0] arp_table_source_payload_ip_address;
reg [47:0] arp_table_source_payload_mac_address;
reg arp_table_request_request_valid;
reg arp_table_request_request_ready;
wire [31:0] arp_table_request_request_payload_ip_address;
reg arp_table_response_response_valid;
reg arp_table_response_response_ready;
reg arp_table_response_response_payload_failed = 1'd0;
reg [47:0] arp_table_response_response_payload_mac_address = 48'd0;
reg arp_table_request_pending = 1'd0;
reg [3:0] arp_table_request_counter = 4'd0;
reg [31:0] arp_table_request_ip_address = 32'd0;
wire arp_table_request_timer_wait;
wire arp_table_request_timer_done;
reg [22:0] arp_table_request_timer_count = 23'd5000000;
reg arp_table_cache_update_valid;
reg arp_table_cache_update_ready;
reg [31:0] arp_table_cache_update_payload_ip_address;
reg [47:0] arp_table_cache_update_payload_mac_address;
reg arp_table_cache_request_valid;
reg arp_table_cache_request_ready;
reg [31:0] arp_table_cache_request_payload_ip_address;
reg arp_table_cache_response_valid;
reg [47:0] arp_table_cache_response_payload_mac_address;
reg arp_table_cache_response_payload_error;
reg arp_table_cache_enable = 1'd1;
reg arp_table_cache_clear_enable = 1'd1;
reg arp_table_cache_update_count = 1'd0;
reg arp_table_cache_search_count = 1'd0;
reg arp_table_cache_error = 1'd0;
reg arp_table_cache_mem_wr_port_adr;
wire [80:0] arp_table_cache_mem_wr_port_dat_r;
reg arp_table_cache_mem_wr_port_we;
reg [80:0] arp_table_cache_mem_wr_port_dat_w;
reg arp_table_cache_mem_rd_port_adr;
wire [80:0] arp_table_cache_mem_rd_port_dat_r;
reg arp_table_cache_mem_wr_port_valid;
reg [31:0] arp_table_cache_mem_wr_port_ip_address;
reg [47:0] arp_table_cache_mem_wr_port_mac_address;
wire arp_table_cache_mem_rd_port_valid;
wire [31:0] arp_table_cache_mem_rd_port_ip_address;
wire [47:0] arp_table_cache_mem_rd_port_mac_address;
reg arp_table_cache_wait;
wire arp_table_cache_done;
reg [25:0] arp_table_cache_count = 26'd50000000;
wire arp_mac_port_sink_valid;
reg arp_mac_port_sink_ready;
wire arp_mac_port_sink_first;
wire arp_mac_port_sink_last;
wire [15:0] arp_mac_port_sink_payload_ethernet_type;
wire [47:0] arp_mac_port_sink_payload_sender_mac;
wire [47:0] arp_mac_port_sink_payload_target_mac;
wire [7:0] arp_mac_port_sink_payload_data;
wire arp_mac_port_sink_payload_be;
wire arp_mac_port_sink_payload_error;
reg arp_mac_port_source_valid;
wire arp_mac_port_source_ready;
reg arp_mac_port_source_first;
reg arp_mac_port_source_last;
reg [15:0] arp_mac_port_source_payload_ethernet_type;
reg [47:0] arp_mac_port_source_payload_sender_mac;
reg [47:0] arp_mac_port_source_payload_target_mac;
reg [7:0] arp_mac_port_source_payload_data;
reg arp_mac_port_source_payload_be;
reg arp_mac_port_source_payload_error;
wire ip_tx_sink_sink_valid0;
wire ip_tx_sink_sink_ready0;
wire ip_tx_sink_sink_first0;
wire ip_tx_sink_sink_last0;
wire [7:0] ip_tx_sink_sink_payload_data0;
wire ip_tx_sink_sink_payload_be0;
wire ip_tx_sink_sink_payload_error0;
wire [15:0] ip_tx_sink_sink_param_length0;
wire [7:0] ip_tx_sink_sink_param_protocol0;
wire [31:0] ip_tx_sink_sink_param_ip_address0;
wire ip_tx_source_source_valid0;
wire ip_tx_source_source_ready0;
wire ip_tx_source_source_first0;
wire ip_tx_source_source_last0;
wire [15:0] ip_tx_source_source_payload_ethernet_type;
wire [47:0] ip_tx_source_source_payload_sender_mac;
wire [47:0] ip_tx_source_source_payload_target_mac;
wire [7:0] ip_tx_source_source_payload_data0;
wire ip_tx_source_source_payload_be0;
wire ip_tx_source_source_payload_error0;
reg ip_tx_target_unreachable;
wire ip_tx_sink_sink_valid1;
wire ip_tx_sink_sink_ready1;
wire ip_tx_sink_sink_first1;
wire ip_tx_sink_sink_last1;
wire [7:0] ip_tx_sink_sink_payload_data1;
wire ip_tx_sink_sink_payload_be1;
wire ip_tx_sink_sink_payload_error1;
wire [15:0] ip_tx_sink_sink_param_length1;
wire [7:0] ip_tx_sink_sink_param_protocol1;
wire [31:0] ip_tx_sink_sink_param_ip_address1;
wire ip_tx_source_source_valid1;
wire ip_tx_source_source_ready1;
wire ip_tx_source_source_first1;
wire ip_tx_source_source_last1;
wire [7:0] ip_tx_source_source_payload_data1;
wire ip_tx_source_source_payload_be1;
wire ip_tx_source_source_payload_error1;
wire [15:0] ip_tx_source_source_param_length;
wire [7:0] ip_tx_source_source_param_protocol;
wire [31:0] ip_tx_source_source_param_ip_address;
wire ip_tx_pipe_valid_sink_valid;
wire ip_tx_pipe_valid_sink_ready;
wire ip_tx_pipe_valid_sink_first;
wire ip_tx_pipe_valid_sink_last;
wire [7:0] ip_tx_pipe_valid_sink_payload_data;
wire ip_tx_pipe_valid_sink_payload_be;
wire ip_tx_pipe_valid_sink_payload_error;
wire [15:0] ip_tx_pipe_valid_sink_param_length;
wire [7:0] ip_tx_pipe_valid_sink_param_protocol;
wire [31:0] ip_tx_pipe_valid_sink_param_ip_address;
reg ip_tx_pipe_valid_source_valid = 1'd0;
wire ip_tx_pipe_valid_source_ready;
reg ip_tx_pipe_valid_source_first = 1'd0;
reg ip_tx_pipe_valid_source_last = 1'd0;
reg [7:0] ip_tx_pipe_valid_source_payload_data = 8'd0;
reg ip_tx_pipe_valid_source_payload_be = 1'd0;
reg ip_tx_pipe_valid_source_payload_error = 1'd0;
reg [15:0] ip_tx_pipe_valid_source_param_length = 16'd0;
reg [7:0] ip_tx_pipe_valid_source_param_protocol = 8'd0;
reg [31:0] ip_tx_pipe_valid_source_param_ip_address = 32'd0;
wire [159:0] ip_tx_liteethipv4checksum_header;
wire [15:0] ip_tx_liteethipv4checksum_value;
wire ip_tx_liteethipv4checksum_done;
reg [16:0] ip_tx_liteethipv4checksum_r = 17'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next0;
reg [16:0] ip_tx_liteethipv4checksum_r_next0 = 17'd0;
reg ip_tx_liteethipv4checksum0 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next1;
reg [16:0] ip_tx_liteethipv4checksum_r_next1 = 17'd0;
reg ip_tx_liteethipv4checksum1 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next2;
reg [16:0] ip_tx_liteethipv4checksum_r_next2 = 17'd0;
reg ip_tx_liteethipv4checksum2 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next3;
reg [16:0] ip_tx_liteethipv4checksum_r_next3 = 17'd0;
reg ip_tx_liteethipv4checksum3 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next4;
reg [16:0] ip_tx_liteethipv4checksum_r_next4 = 17'd0;
reg ip_tx_liteethipv4checksum4 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next5;
reg [16:0] ip_tx_liteethipv4checksum_r_next5 = 17'd0;
reg ip_tx_liteethipv4checksum5 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next6;
reg [16:0] ip_tx_liteethipv4checksum_r_next6 = 17'd0;
reg ip_tx_liteethipv4checksum6 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next7;
reg [16:0] ip_tx_liteethipv4checksum_r_next7 = 17'd0;
reg ip_tx_liteethipv4checksum7 = 1'd0;
wire [16:0] ip_tx_liteethipv4checksum_s_next8;
reg [16:0] ip_tx_liteethipv4checksum_r_next8 = 17'd0;
reg ip_tx_liteethipv4checksum8 = 1'd0;
reg [3:0] ip_tx_liteethipv4checksum_counter = 4'd0;
wire ip_tx_liteethipv4checksum_counter_ce;
wire ip_tx_ce;
wire ip_tx_reset;
wire ip_tx_packetizer_sink_valid;
reg ip_tx_packetizer_sink_ready;
wire ip_tx_packetizer_sink_last;
wire [7:0] ip_tx_packetizer_sink_payload_data;
wire ip_tx_packetizer_sink_payload_be;
reg ip_tx_packetizer_sink_payload_error = 1'd0;
wire [15:0] ip_tx_packetizer_sink_param_checksum;
wire [15:0] ip_tx_packetizer_sink_param_flags_offset;
wire [15:0] ip_tx_packetizer_sink_param_identification;
wire [3:0] ip_tx_packetizer_sink_param_ihl;
wire [7:0] ip_tx_packetizer_sink_param_protocol;
wire [31:0] ip_tx_packetizer_sink_param_sender_ip;
wire [31:0] ip_tx_packetizer_sink_param_target_ip;
wire [15:0] ip_tx_packetizer_sink_param_total_length;
wire [7:0] ip_tx_packetizer_sink_param_ttl;
wire [3:0] ip_tx_packetizer_sink_param_version;
reg ip_tx_packetizer_source_valid;
wire ip_tx_packetizer_source_ready;
reg ip_tx_packetizer_source_first;
reg ip_tx_packetizer_source_last;
reg [7:0] ip_tx_packetizer_source_payload_data;
wire ip_tx_packetizer_source_payload_be;
reg ip_tx_packetizer_source_payload_error;
reg [159:0] ip_tx_packetizer_header;
reg ip_tx_packetizer_source_be;
reg [4:0] ip_tx_packetizer_count = 5'd0;
reg ip_tx_packetizer_is_ongoing;
reg [47:0] ip_tx_target_mac = 48'd0;
reg ip_tx_fsm_is_ongoing0;
reg ip_tx_fsm_is_ongoing1;
reg ip_tx_fsm_is_ongoing2;
wire ip_rx_sink_sink_valid;
wire ip_rx_sink_sink_ready;
wire ip_rx_sink_sink_first;
wire ip_rx_sink_sink_last;
wire [15:0] ip_rx_sink_sink_payload_ethernet_type;
wire [47:0] ip_rx_sink_sink_payload_sender_mac;
wire [47:0] ip_rx_sink_sink_payload_target_mac;
wire [7:0] ip_rx_sink_sink_payload_data;
wire ip_rx_sink_sink_payload_be;
wire ip_rx_sink_sink_payload_error;
wire ip_rx_source_source_valid;
wire ip_rx_source_source_ready;
reg ip_rx_source_source_first = 1'd0;
wire ip_rx_source_source_last;
wire [7:0] ip_rx_source_source_payload_data;
wire ip_rx_source_source_payload_be;
wire ip_rx_source_source_payload_error;
wire [15:0] ip_rx_source_source_param_length;
wire [7:0] ip_rx_source_source_param_protocol;
wire [31:0] ip_rx_source_source_param_ip_address;
wire ip_rx_depacketizer_sink_valid;
reg ip_rx_depacketizer_sink_ready;
wire ip_rx_depacketizer_sink_first;
wire ip_rx_depacketizer_sink_last;
wire [15:0] ip_rx_depacketizer_sink_payload_ethernet_type;
wire [47:0] ip_rx_depacketizer_sink_payload_sender_mac;
wire [47:0] ip_rx_depacketizer_sink_payload_target_mac;
wire [7:0] ip_rx_depacketizer_sink_payload_data;
wire ip_rx_depacketizer_sink_payload_be;
wire ip_rx_depacketizer_sink_payload_error;
reg ip_rx_depacketizer_source_valid;
wire ip_rx_depacketizer_source_ready;
reg ip_rx_depacketizer_source_first;
reg ip_rx_depacketizer_source_last;
reg [7:0] ip_rx_depacketizer_source_payload_data;
reg ip_rx_depacketizer_source_payload_be;
wire ip_rx_depacketizer_source_payload_error;
wire [15:0] ip_rx_depacketizer_source_param_checksum;
wire [15:0] ip_rx_depacketizer_source_param_flags_offset;
wire [15:0] ip_rx_depacketizer_source_param_identification;
wire [3:0] ip_rx_depacketizer_source_param_ihl;
wire [7:0] ip_rx_depacketizer_source_param_protocol;
wire [31:0] ip_rx_depacketizer_source_param_sender_ip;
wire [31:0] ip_rx_depacketizer_source_param_target_ip;
wire [15:0] ip_rx_depacketizer_source_param_total_length;
wire [7:0] ip_rx_depacketizer_source_param_ttl;
wire [3:0] ip_rx_depacketizer_source_param_version;
wire [159:0] ip_rx_depacketizer_header;
reg [159:0] ip_rx_depacketizer_sr = 160'd0;
reg [4:0] ip_rx_depacketizer_count = 5'd0;
reg ip_rx_depacketizer_skip_cnt = 1'd0;
reg ip_rx_depacketizer_data_copy_first = 1'd1;
wire ip_rx_depacketizer_sr_shift;
reg ip_rx_depacketizer_is_ongoing0;
reg ip_rx_depacketizer_is_ongoing1;
reg ip_rx_depacketizer_send_dummy = 1'd0;
wire [159:0] ip_rx_liteethipv4checksum_header;
wire [15:0] ip_rx_liteethipv4checksum_value;
wire ip_rx_liteethipv4checksum_done;
reg [16:0] ip_rx_liteethipv4checksum_r = 17'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next0;
reg [16:0] ip_rx_liteethipv4checksum_r_next0 = 17'd0;
reg ip_rx_liteethipv4checksum0 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next1;
reg [16:0] ip_rx_liteethipv4checksum_r_next1 = 17'd0;
reg ip_rx_liteethipv4checksum1 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next2;
reg [16:0] ip_rx_liteethipv4checksum_r_next2 = 17'd0;
reg ip_rx_liteethipv4checksum2 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next3;
reg [16:0] ip_rx_liteethipv4checksum_r_next3 = 17'd0;
reg ip_rx_liteethipv4checksum3 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next4;
reg [16:0] ip_rx_liteethipv4checksum_r_next4 = 17'd0;
reg ip_rx_liteethipv4checksum4 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next5;
reg [16:0] ip_rx_liteethipv4checksum_r_next5 = 17'd0;
reg ip_rx_liteethipv4checksum5 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next6;
reg [16:0] ip_rx_liteethipv4checksum_r_next6 = 17'd0;
reg ip_rx_liteethipv4checksum6 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next7;
reg [16:0] ip_rx_liteethipv4checksum_r_next7 = 17'd0;
reg ip_rx_liteethipv4checksum7 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next8;
reg [16:0] ip_rx_liteethipv4checksum_r_next8 = 17'd0;
reg ip_rx_liteethipv4checksum8 = 1'd0;
wire [16:0] ip_rx_liteethipv4checksum_s_next9;
reg [16:0] ip_rx_liteethipv4checksum_r_next9 = 17'd0;
reg ip_rx_liteethipv4checksum9 = 1'd0;
reg [3:0] ip_rx_liteethipv4checksum_counter = 4'd0;
wire ip_rx_liteethipv4checksum_counter_ce;
wire ip_rx_ce;
wire ip_rx_reset;
wire ip_rx_accepted;
wire ip_rx_receiving;
reg ip_rx_fsm_is_ongoing0;
reg ip_rx_fsm_is_ongoing1;
wire ip_mac_port_sink_valid;
reg ip_mac_port_sink_ready;
wire ip_mac_port_sink_first;
wire ip_mac_port_sink_last;
wire [15:0] ip_mac_port_sink_payload_ethernet_type;
wire [47:0] ip_mac_port_sink_payload_sender_mac;
wire [47:0] ip_mac_port_sink_payload_target_mac;
wire [7:0] ip_mac_port_sink_payload_data;
wire ip_mac_port_sink_payload_be;
wire ip_mac_port_sink_payload_error;
reg ip_mac_port_source_valid;
wire ip_mac_port_source_ready;
reg ip_mac_port_source_first;
reg ip_mac_port_source_last;
reg [15:0] ip_mac_port_source_payload_ethernet_type;
reg [47:0] ip_mac_port_source_payload_sender_mac;
reg [47:0] ip_mac_port_source_payload_target_mac;
reg [7:0] ip_mac_port_source_payload_data;
reg ip_mac_port_source_payload_be;
reg ip_mac_port_source_payload_error;
wire ip_rx_buffer_sink_sink_valid;
wire ip_rx_buffer_sink_sink_ready;
wire ip_rx_buffer_sink_sink_first;
wire ip_rx_buffer_sink_sink_last;
wire [7:0] ip_rx_buffer_sink_sink_payload_data;
wire ip_rx_buffer_sink_sink_payload_be;
wire ip_rx_buffer_sink_sink_payload_error;
wire [15:0] ip_rx_buffer_sink_sink_param_length;
wire [7:0] ip_rx_buffer_sink_sink_param_protocol;
wire [31:0] ip_rx_buffer_sink_sink_param_ip_address;
wire ip_rx_buffer_source_source_valid;
wire ip_rx_buffer_source_source_ready;
wire ip_rx_buffer_source_source_first;
wire ip_rx_buffer_source_source_last;
wire [7:0] ip_rx_buffer_source_source_payload_data;
wire ip_rx_buffer_source_source_payload_be;
wire ip_rx_buffer_source_source_payload_error;
wire [15:0] ip_rx_buffer_source_source_param_length;
wire [7:0] ip_rx_buffer_source_source_param_protocol;
wire [31:0] ip_rx_buffer_source_source_param_ip_address;
wire ip_rx_buffer_pipe_valid_sink_valid;
wire ip_rx_buffer_pipe_valid_sink_ready;
wire ip_rx_buffer_pipe_valid_sink_first;
wire ip_rx_buffer_pipe_valid_sink_last;
wire [7:0] ip_rx_buffer_pipe_valid_sink_payload_data;
wire ip_rx_buffer_pipe_valid_sink_payload_be;
wire ip_rx_buffer_pipe_valid_sink_payload_error;
wire [15:0] ip_rx_buffer_pipe_valid_sink_param_length;
wire [7:0] ip_rx_buffer_pipe_valid_sink_param_protocol;
wire [31:0] ip_rx_buffer_pipe_valid_sink_param_ip_address;
reg ip_rx_buffer_pipe_valid_source_valid = 1'd0;
wire ip_rx_buffer_pipe_valid_source_ready;
reg ip_rx_buffer_pipe_valid_source_first = 1'd0;
reg ip_rx_buffer_pipe_valid_source_last = 1'd0;
reg [7:0] ip_rx_buffer_pipe_valid_source_payload_data = 8'd0;
reg ip_rx_buffer_pipe_valid_source_payload_be = 1'd0;
reg ip_rx_buffer_pipe_valid_source_payload_error = 1'd0;
reg [15:0] ip_rx_buffer_pipe_valid_source_param_length = 16'd0;
reg [7:0] ip_rx_buffer_pipe_valid_source_param_protocol = 8'd0;
reg [31:0] ip_rx_buffer_pipe_valid_source_param_ip_address = 32'd0;
wire ip_rx_buffer_pipe_ready_sink_valid;
wire ip_rx_buffer_pipe_ready_sink_ready;
wire ip_rx_buffer_pipe_ready_sink_first;
wire ip_rx_buffer_pipe_ready_sink_last;
wire [7:0] ip_rx_buffer_pipe_ready_sink_payload_data;
wire ip_rx_buffer_pipe_ready_sink_payload_be;
wire ip_rx_buffer_pipe_ready_sink_payload_error;
wire [15:0] ip_rx_buffer_pipe_ready_sink_param_length;
wire [7:0] ip_rx_buffer_pipe_ready_sink_param_protocol;
wire [31:0] ip_rx_buffer_pipe_ready_sink_param_ip_address;
reg ip_rx_buffer_pipe_ready_source_valid;
wire ip_rx_buffer_pipe_ready_source_ready;
reg ip_rx_buffer_pipe_ready_source_first;
reg ip_rx_buffer_pipe_ready_source_last;
reg [7:0] ip_rx_buffer_pipe_ready_source_payload_data;
reg ip_rx_buffer_pipe_ready_source_payload_be;
reg ip_rx_buffer_pipe_ready_source_payload_error;
reg [15:0] ip_rx_buffer_pipe_ready_source_param_length;
reg [7:0] ip_rx_buffer_pipe_ready_source_param_protocol;
reg [31:0] ip_rx_buffer_pipe_ready_source_param_ip_address;
reg ip_rx_buffer_pipe_ready_valid = 1'd0;
reg ip_rx_buffer_pipe_ready_sink_d_valid = 1'd0;
reg ip_rx_buffer_pipe_ready_sink_d_ready = 1'd0;
reg ip_rx_buffer_pipe_ready_sink_d_first = 1'd0;
reg ip_rx_buffer_pipe_ready_sink_d_last = 1'd0;
reg [7:0] ip_rx_buffer_pipe_ready_sink_d_payload_data = 8'd0;
reg ip_rx_buffer_pipe_ready_sink_d_payload_be = 1'd0;
reg ip_rx_buffer_pipe_ready_sink_d_payload_error = 1'd0;
reg [15:0] ip_rx_buffer_pipe_ready_sink_d_param_length = 16'd0;
reg [7:0] ip_rx_buffer_pipe_ready_sink_d_param_protocol = 8'd0;
reg [31:0] ip_rx_buffer_pipe_ready_sink_d_param_ip_address = 32'd0;
reg ip_crossbar_source_valid;
wire ip_crossbar_source_ready;
reg ip_crossbar_source_first;
reg ip_crossbar_source_last;
reg [7:0] ip_crossbar_source_payload_data;
reg ip_crossbar_source_payload_be;
reg ip_crossbar_source_payload_error;
reg [15:0] ip_crossbar_source_param_length;
reg [7:0] ip_crossbar_source_param_protocol;
reg [31:0] ip_crossbar_source_param_ip_address;
wire ip_crossbar_sink_valid;
reg ip_crossbar_sink_ready;
wire ip_crossbar_sink_first;
wire ip_crossbar_sink_last;
wire [7:0] ip_crossbar_sink_payload_data;
wire ip_crossbar_sink_payload_be;
wire ip_crossbar_sink_payload_error;
wire [15:0] ip_crossbar_sink_param_length;
wire [7:0] ip_crossbar_sink_param_protocol;
wire [31:0] ip_crossbar_sink_param_ip_address;
wire icmp_tx_sink_sink_valid;
wire icmp_tx_sink_sink_ready;
wire icmp_tx_sink_sink_first;
wire icmp_tx_sink_sink_last;
wire [7:0] icmp_tx_sink_sink_payload_data;
wire icmp_tx_sink_sink_payload_be;
wire icmp_tx_sink_sink_payload_error;
wire [15:0] icmp_tx_sink_sink_param_checksum;
wire [7:0] icmp_tx_sink_sink_param_code;
wire [7:0] icmp_tx_sink_sink_param_msgtype;
wire [31:0] icmp_tx_sink_sink_param_quench;
wire [31:0] icmp_tx_sink_sink_param_ip_address;
wire [15:0] icmp_tx_sink_sink_param_length;
wire icmp_tx_source_source_valid;
wire icmp_tx_source_source_ready;
wire icmp_tx_source_source_first;
wire icmp_tx_source_source_last;
wire [7:0] icmp_tx_source_source_payload_data;
wire icmp_tx_source_source_payload_be;
wire icmp_tx_source_source_payload_error;
wire [15:0] icmp_tx_source_source_param_length;
wire [7:0] icmp_tx_source_source_param_protocol;
wire [31:0] icmp_tx_source_source_param_ip_address;
wire icmp_tx_packetizer_sink_valid;
reg icmp_tx_packetizer_sink_ready;
wire icmp_tx_packetizer_sink_last;
wire [7:0] icmp_tx_packetizer_sink_payload_data;
wire icmp_tx_packetizer_sink_payload_be;
reg icmp_tx_packetizer_sink_payload_error = 1'd0;
wire [15:0] icmp_tx_packetizer_sink_param_checksum;
wire [7:0] icmp_tx_packetizer_sink_param_code;
wire [7:0] icmp_tx_packetizer_sink_param_msgtype;
wire [31:0] icmp_tx_packetizer_sink_param_quench;
reg icmp_tx_packetizer_source_valid;
wire icmp_tx_packetizer_source_ready;
reg icmp_tx_packetizer_source_first;
reg icmp_tx_packetizer_source_last;
reg [7:0] icmp_tx_packetizer_source_payload_data;
wire icmp_tx_packetizer_source_payload_be;
reg icmp_tx_packetizer_source_payload_error;
reg [63:0] icmp_tx_packetizer_header;
reg icmp_tx_packetizer_source_be;
reg [2:0] icmp_tx_packetizer_count = 3'd0;
reg icmp_tx_packetizer_is_ongoing;
reg icmp_tx_active = 1'd0;
reg [15:0] icmp_tx_length = 16'd0;
reg [31:0] icmp_tx_ip_address = 32'd0;
wire icmp_rx_sink_sink_valid;
wire icmp_rx_sink_sink_ready;
wire icmp_rx_sink_sink_first;
wire icmp_rx_sink_sink_last;
wire [7:0] icmp_rx_sink_sink_payload_data;
wire icmp_rx_sink_sink_payload_be;
wire icmp_rx_sink_sink_payload_error;
wire [15:0] icmp_rx_sink_sink_param_length;
wire [7:0] icmp_rx_sink_sink_param_protocol;
wire [31:0] icmp_rx_sink_sink_param_ip_address;
wire icmp_rx_source_source_valid;
wire icmp_rx_source_source_ready;
reg icmp_rx_source_source_first = 1'd0;
reg icmp_rx_source_source_last;
wire [7:0] icmp_rx_source_source_payload_data;
reg icmp_rx_source_source_payload_be;
reg icmp_rx_source_source_payload_error;
wire [15:0] icmp_rx_source_source_param_checksum;
wire [7:0] icmp_rx_source_source_param_code;
wire [7:0] icmp_rx_source_source_param_msgtype;
wire [31:0] icmp_rx_source_source_param_quench;
wire [31:0] icmp_rx_source_source_param_ip_address;
wire [15:0] icmp_rx_source_source_param_length;
wire icmp_rx_depacketizer_sink_valid;
reg icmp_rx_depacketizer_sink_ready;
wire icmp_rx_depacketizer_sink_first;
wire icmp_rx_depacketizer_sink_last;
wire [7:0] icmp_rx_depacketizer_sink_payload_data;
wire icmp_rx_depacketizer_sink_payload_be;
wire icmp_rx_depacketizer_sink_payload_error;
wire [15:0] icmp_rx_depacketizer_sink_param_length;
wire [7:0] icmp_rx_depacketizer_sink_param_protocol;
wire [31:0] icmp_rx_depacketizer_sink_param_ip_address;
reg icmp_rx_depacketizer_source_valid;
wire icmp_rx_depacketizer_source_ready;
reg icmp_rx_depacketizer_source_first;
reg icmp_rx_depacketizer_source_last;
reg [7:0] icmp_rx_depacketizer_source_payload_data;
wire icmp_rx_depacketizer_source_payload_be;
wire icmp_rx_depacketizer_source_payload_error;
wire [15:0] icmp_rx_depacketizer_source_param_checksum;
wire [7:0] icmp_rx_depacketizer_source_param_code;
wire [7:0] icmp_rx_depacketizer_source_param_msgtype;
wire [31:0] icmp_rx_depacketizer_source_param_quench;
wire [63:0] icmp_rx_depacketizer_header;
reg [63:0] icmp_rx_depacketizer_sr = 64'd0;
reg icmp_rx_depacketizer_sr_shift;
reg icmp_rx_depacketizer_sr_shift_leftover = 1'd0;
reg [2:0] icmp_rx_depacketizer_count = 3'd0;
reg icmp_rx_depacketizer_data_copy_first = 1'd0;
wire icmp_rx_depacketizer_sink_be;
reg icmp_rx_depacketizer_source_be;
reg icmp_rx_depacketizer_fsm_from_idle = 1'd0;
reg icmp_rx_active = 1'd0;
reg [15:0] icmp_rx_length = 16'd0;
reg [7:0] icmp_rx_protocol = 8'd0;
reg [31:0] icmp_rx_ip_address = 32'd0;
reg [16:0] icmp_rx_count = 17'd0;
reg icmp_rx_is_ongoing0;
reg icmp_rx_is_ongoing1;
reg icmp_rx_is_ongoing2;
wire icmp_echo_sink_sink_valid;
wire icmp_echo_sink_sink_ready;
wire icmp_echo_sink_sink_first;
wire icmp_echo_sink_sink_last;
wire [7:0] icmp_echo_sink_sink_payload_data;
wire icmp_echo_sink_sink_payload_be;
wire icmp_echo_sink_sink_payload_error;
wire [15:0] icmp_echo_sink_sink_param_checksum;
wire [7:0] icmp_echo_sink_sink_param_code;
wire [7:0] icmp_echo_sink_sink_param_msgtype;
wire [31:0] icmp_echo_sink_sink_param_quench;
wire [31:0] icmp_echo_sink_sink_param_ip_address;
wire [15:0] icmp_echo_sink_sink_param_length;
wire icmp_echo_source_source_valid;
wire icmp_echo_source_source_ready;
wire icmp_echo_source_source_first;
wire icmp_echo_source_source_last;
wire [7:0] icmp_echo_source_source_payload_data;
wire icmp_echo_source_source_payload_be;
wire icmp_echo_source_source_payload_error;
wire [15:0] icmp_echo_source_source_param_checksum;
wire [7:0] icmp_echo_source_source_param_code;
reg [7:0] icmp_echo_source_source_param_msgtype;
wire [31:0] icmp_echo_source_source_param_quench;
wire [31:0] icmp_echo_source_source_param_ip_address;
wire [15:0] icmp_echo_source_source_param_length;
wire icmp_echo_buffer_sink_valid;
wire icmp_echo_buffer_sink_ready;
wire icmp_echo_buffer_sink_first;
wire icmp_echo_buffer_sink_last;
wire [7:0] icmp_echo_buffer_sink_payload_data;
wire icmp_echo_buffer_sink_payload_be;
wire icmp_echo_buffer_sink_payload_error;
wire [15:0] icmp_echo_buffer_sink_param_checksum;
wire [7:0] icmp_echo_buffer_sink_param_code;
wire [7:0] icmp_echo_buffer_sink_param_msgtype;
wire [31:0] icmp_echo_buffer_sink_param_quench;
wire [31:0] icmp_echo_buffer_sink_param_ip_address;
wire [15:0] icmp_echo_buffer_sink_param_length;
wire icmp_echo_buffer_source_valid;
wire icmp_echo_buffer_source_ready;
wire icmp_echo_buffer_source_first;
wire icmp_echo_buffer_source_last;
wire [7:0] icmp_echo_buffer_source_payload_data;
wire icmp_echo_buffer_source_payload_be;
wire icmp_echo_buffer_source_payload_error;
wire [15:0] icmp_echo_buffer_source_param_checksum;
wire [7:0] icmp_echo_buffer_source_param_code;
wire [7:0] icmp_echo_buffer_source_param_msgtype;
wire [31:0] icmp_echo_buffer_source_param_quench;
wire [31:0] icmp_echo_buffer_source_param_ip_address;
wire [15:0] icmp_echo_buffer_source_param_length;
wire icmp_echo_buffer_discard;
wire icmp_echo_buffer_drop;
wire icmp_echo_buffer_sink_sink_valid;
wire icmp_echo_buffer_sink_sink_ready;
reg icmp_echo_buffer_sink_sink_first = 1'd0;
wire icmp_echo_buffer_sink_sink_last;
wire [15:0] icmp_echo_buffer_sink_sink_param_checksum;
wire [7:0] icmp_echo_buffer_sink_sink_param_code;
wire [7:0] icmp_echo_buffer_sink_sink_param_msgtype;
wire [31:0] icmp_echo_buffer_sink_sink_param_quench;
wire [31:0] icmp_echo_buffer_sink_sink_param_ip_address;
wire [15:0] icmp_echo_buffer_sink_sink_param_length;
wire icmp_echo_buffer_source_source_valid;
wire icmp_echo_buffer_source_source_ready;
wire icmp_echo_buffer_source_source_first;
wire icmp_echo_buffer_source_source_last;
wire [15:0] icmp_echo_buffer_source_source_param_checksum;
wire [7:0] icmp_echo_buffer_source_source_param_code;
wire [7:0] icmp_echo_buffer_source_source_param_msgtype;
wire [31:0] icmp_echo_buffer_source_source_param_quench;
wire [31:0] icmp_echo_buffer_source_source_param_ip_address;
wire [15:0] icmp_echo_buffer_source_source_param_length;
wire icmp_echo_buffer_pipe_valid_sink_valid;
wire icmp_echo_buffer_pipe_valid_sink_ready;
wire icmp_echo_buffer_pipe_valid_sink_first;
wire icmp_echo_buffer_pipe_valid_sink_last;
wire [15:0] icmp_echo_buffer_pipe_valid_sink_param_checksum;
wire [7:0] icmp_echo_buffer_pipe_valid_sink_param_code;
wire [7:0] icmp_echo_buffer_pipe_valid_sink_param_msgtype;
wire [31:0] icmp_echo_buffer_pipe_valid_sink_param_quench;
wire [31:0] icmp_echo_buffer_pipe_valid_sink_param_ip_address;
wire [15:0] icmp_echo_buffer_pipe_valid_sink_param_length;
reg icmp_echo_buffer_pipe_valid_source_valid = 1'd0;
wire icmp_echo_buffer_pipe_valid_source_ready;
reg icmp_echo_buffer_pipe_valid_source_first = 1'd0;
reg icmp_echo_buffer_pipe_valid_source_last = 1'd0;
reg [15:0] icmp_echo_buffer_pipe_valid_source_param_checksum = 16'd0;
reg [7:0] icmp_echo_buffer_pipe_valid_source_param_code = 8'd0;
reg [7:0] icmp_echo_buffer_pipe_valid_source_param_msgtype = 8'd0;
reg [31:0] icmp_echo_buffer_pipe_valid_source_param_quench = 32'd0;
reg [31:0] icmp_echo_buffer_pipe_valid_source_param_ip_address = 32'd0;
reg [15:0] icmp_echo_buffer_pipe_valid_source_param_length = 16'd0;
wire icmp_echo_buffer_level;
wire [6:0] icmp_echo_buffer_wr_port_adr;
wire [11:0] icmp_echo_buffer_wr_port_dat_r;
wire icmp_echo_buffer_wr_port_we;
wire [11:0] icmp_echo_buffer_wr_port_dat_w;
wire [6:0] icmp_echo_buffer_rd_port_adr;
wire [11:0] icmp_echo_buffer_rd_port_dat_r;
wire icmp_echo_buffer_rd_port_re;
reg [7:0] icmp_echo_buffer_wr_ptr = 8'd0;
reg [7:0] icmp_echo_buffer_commit_ptr = 8'd0;
reg [7:0] icmp_echo_buffer_rd_ptr = 8'd0;
reg icmp_echo_buffer_out_valid = 1'd0;
wire [7:0] icmp_echo_buffer_used;
wire icmp_echo_buffer_full;
reg icmp_echo_buffer_overflow = 1'd0;
reg icmp_echo_buffer_invalid = 1'd0;
wire icmp_echo_buffer_commit;
wire icmp_echo_buffer_fetch;
reg icmp_echo_busy = 1'd0;
reg icmp_echo_receiving = 1'd0;
reg [7:0] icmp_echo_count = 8'd0;
wire [7:0] icmp_echo_size;
wire icmp_sink_valid;
reg icmp_sink_ready;
wire icmp_sink_first;
wire icmp_sink_last;
wire [7:0] icmp_sink_payload_data;
wire icmp_sink_payload_be;
wire icmp_sink_payload_error;
wire [15:0] icmp_sink_param_length;
wire [7:0] icmp_sink_param_protocol;
wire [31:0] icmp_sink_param_ip_address;
reg icmp_source_valid;
wire icmp_source_ready;
reg icmp_source_first;
reg icmp_source_last;
reg [7:0] icmp_source_payload_data;
reg icmp_source_payload_be;
reg icmp_source_payload_error;
reg [15:0] icmp_source_param_length;
reg [7:0] icmp_source_param_protocol;
reg [31:0] icmp_source_param_ip_address;
wire tx_sink_sink_valid;
wire tx_sink_sink_ready;
wire tx_sink_sink_first;
wire tx_sink_sink_last;
wire [7:0] tx_sink_sink_payload_data;
wire tx_sink_sink_payload_be;
wire tx_sink_sink_payload_error;
wire [15:0] tx_sink_sink_param_src_port;
wire [15:0] tx_sink_sink_param_dst_port;
wire [31:0] tx_sink_sink_param_ip_address;
wire [15:0] tx_sink_sink_param_length;
wire tx_source_source_valid;
wire tx_source_source_ready;
wire tx_source_source_first;
wire tx_source_source_last;
wire [7:0] tx_source_source_payload_data;
wire tx_source_source_payload_be;
wire tx_source_source_payload_error;
wire [15:0] tx_source_source_param_length;
wire [7:0] tx_source_source_param_protocol;
wire [31:0] tx_source_source_param_ip_address;
wire tx_packetizer_sink_valid;
reg tx_packetizer_sink_ready;
wire tx_packetizer_sink_last;
wire [7:0] tx_packetizer_sink_payload_data;
wire tx_packetizer_sink_payload_be;
reg tx_packetizer_sink_payload_error = 1'd0;
wire [15:0] tx_packetizer_sink_param_checksum;
wire [15:0] tx_packetizer_sink_param_dst_port;
wire [15:0] tx_packetizer_sink_param_length;
wire [15:0] tx_packetizer_sink_param_src_port;
reg tx_packetizer_source_valid;
wire tx_packetizer_source_ready;
reg tx_packetizer_source_first;
reg tx_packetizer_source_last;
reg [7:0] tx_packetizer_source_payload_data;
wire tx_packetizer_source_payload_be;
reg tx_packetizer_source_payload_error;
reg [63:0] tx_packetizer_header;
reg tx_packetizer_source_be;
reg [2:0] tx_packetizer_count = 3'd0;
reg tx_packetizer_is_ongoing;
reg tx_active = 1'd0;
reg [15:0] tx_length = 16'd0;
reg [31:0] tx_ip_address = 32'd0;
wire rx_sink_sink_valid;
wire rx_sink_sink_ready;
wire rx_sink_sink_first;
wire rx_sink_sink_last;
wire [7:0] rx_sink_sink_payload_data;
wire rx_sink_sink_payload_be;
wire rx_sink_sink_payload_error;
wire [15:0] rx_sink_sink_param_length;
wire [7:0] rx_sink_sink_param_protocol;
wire [31:0] rx_sink_sink_param_ip_address;
wire rx_source_source_valid;
wire rx_source_source_ready;
reg rx_source_source_first = 1'd0;
wire rx_source_source_last;
wire [7:0] rx_source_source_payload_data;
reg rx_source_source_payload_be;
reg rx_source_source_payload_error;
wire [15:0] rx_source_source_param_src_port;
wire [15:0] rx_source_source_param_dst_port;
wire [31:0] rx_source_source_param_ip_address;
wire [15:0] rx_source_source_param_length;
wire rx_depacketizer_sink_valid;
reg rx_depacketizer_sink_ready;
wire rx_depacketizer_sink_first;
wire rx_depacketizer_sink_last;
wire [7:0] rx_depacketizer_sink_payload_data;
wire rx_depacketizer_sink_payload_be;
wire rx_depacketizer_sink_payload_error;
wire [15:0] rx_depacketizer_sink_param_length;
wire [7:0] rx_depacketizer_sink_param_protocol;
wire [31:0] rx_depacketizer_sink_param_ip_address;
reg rx_depacketizer_source_valid;
wire rx_depacketizer_source_ready;
reg rx_depacketizer_source_first;
reg rx_depacketizer_source_last;
reg [7:0] rx_depacketizer_source_payload_data;
reg rx_depacketizer_source_payload_be;
wire rx_depacketizer_source_payload_error;
wire [15:0] rx_depacketizer_source_param_checksum;
wire [15:0] rx_depacketizer_source_param_dst_port;
wire [15:0] rx_depacketizer_source_param_length;
wire [15:0] rx_depacketizer_source_param_src_port;
wire [63:0] rx_depacketizer_header;
reg [63:0] rx_depacketizer_sr = 64'd0;
reg [2:0] rx_depacketizer_count = 3'd0;
reg rx_depacketizer_skip_cnt = 1'd0;
reg rx_depacketizer_data_copy_first = 1'd1;
wire rx_depacketizer_sr_shift;
reg rx_depacketizer_is_ongoing0;
reg rx_depacketizer_is_ongoing1;
reg rx_depacketizer_send_dummy = 1'd0;
reg rx_active = 1'd0;
reg [15:0] rx_length = 16'd0;
reg [7:0] rx_protocol = 8'd0;
reg [31:0] rx_ip_address = 32'd0;
reg [16:0] rx_count = 17'd0;
wire rx_accepted;
wire rx_receiving;
wire [16:0] rx_end_count;
reg rx_fsm_is_ongoing0;
reg rx_fsm_is_ongoing1;
wire ip_port_sink_valid;
reg ip_port_sink_ready;
wire ip_port_sink_first;
wire ip_port_sink_last;
wire [7:0] ip_port_sink_payload_data;
wire ip_port_sink_payload_be;
wire ip_port_sink_payload_error;
wire [15:0] ip_port_sink_param_length;
wire [7:0] ip_port_sink_param_protocol;
wire [31:0] ip_port_sink_param_ip_address;
reg ip_port_source_valid;
wire ip_port_source_ready;
reg ip_port_source_first;
reg ip_port_source_last;
reg [7:0] ip_port_source_payload_data;
reg ip_port_source_payload_be;
reg ip_port_source_payload_error;
reg [15:0] ip_port_source_param_length;
reg [7:0] ip_port_source_param_protocol;
reg [31:0] ip_port_source_param_ip_address;
wire crossbar_source_valid;
wire crossbar_source_ready;
wire crossbar_source_first;
wire crossbar_source_last;
wire [7:0] crossbar_source_payload_data;
wire crossbar_source_payload_be;
wire crossbar_source_payload_error;
wire [15:0] crossbar_source_param_src_port;
wire [15:0] crossbar_source_param_dst_port;
wire [31:0] crossbar_source_param_ip_address;
wire [15:0] crossbar_source_param_length;
wire crossbar_sink_valid;
reg crossbar_sink_ready;
wire crossbar_sink_first;
wire crossbar_sink_last;
wire [7:0] crossbar_sink_payload_data;
wire crossbar_sink_payload_be;
wire crossbar_sink_payload_error;
wire [15:0] crossbar_sink_param_src_port;
wire [15:0] crossbar_sink_param_dst_port;
wire [31:0] crossbar_sink_param_ip_address;
wire [15:0] crossbar_sink_param_length;
wire user_port_sink_valid;
wire user_port_sink_ready;
wire user_port_sink_first;
wire user_port_sink_last;
wire [7:0] user_port_sink_payload_data;
wire user_port_sink_payload_be;
wire user_port_sink_payload_error;
wire [15:0] user_port_sink_param_src_port;
wire [15:0] user_port_sink_param_dst_port;
wire [31:0] user_port_sink_param_ip_address;
wire [15:0] user_port_sink_param_length;
wire user_port_source_valid;
wire user_port_source_ready;
wire user_port_source_first;
wire user_port_source_last;
wire [7:0] user_port_source_payload_data;
wire user_port_source_payload_be;
wire user_port_source_payload_error;
wire [15:0] user_port_source_param_src_port;
wire [15:0] user_port_source_param_dst_port;
wire [31:0] user_port_source_param_ip_address;
wire [15:0] user_port_source_param_length;
wire internal_port_sink_valid;
wire internal_port_sink_ready;
wire internal_port_sink_first;
wire internal_port_sink_last;
wire [7:0] internal_port_sink_payload_data;
wire internal_port_sink_payload_be;
wire internal_port_sink_payload_error;
wire [15:0] internal_port_sink_param_src_port;
wire [15:0] internal_port_sink_param_dst_port;
wire [31:0] internal_port_sink_param_ip_address;
wire [15:0] internal_port_sink_param_length;
reg internal_port_source_valid;
wire internal_port_source_ready;
reg internal_port_source_first;
reg internal_port_source_last;
reg [7:0] internal_port_source_payload_data;
reg internal_port_source_payload_be;
reg internal_port_source_payload_error;
reg [15:0] internal_port_source_param_src_port;
reg [15:0] internal_port_source_param_dst_port;
reg [31:0] internal_port_source_param_ip_address;
reg [15:0] internal_port_source_param_length;
wire tx_cdc_sink_valid;
wire tx_cdc_sink_ready;
wire tx_cdc_sink_first;
wire tx_cdc_sink_last;
wire [7:0] tx_cdc_sink_payload_data;
wire tx_cdc_sink_payload_be;
wire tx_cdc_sink_payload_error;
wire [15:0] tx_cdc_sink_param_src_port;
wire [15:0] tx_cdc_sink_param_dst_port;
wire [31:0] tx_cdc_sink_param_ip_address;
wire [15:0] tx_cdc_sink_param_length;
wire tx_cdc_source_valid;
wire tx_cdc_source_ready;
wire tx_cdc_source_first;
wire tx_cdc_source_last;
wire [7:0] tx_cdc_source_payload_data;
wire tx_cdc_source_payload_be;
wire tx_cdc_source_payload_error;
wire [15:0] tx_cdc_source_param_src_port;
wire [15:0] tx_cdc_source_param_dst_port;
wire [31:0] tx_cdc_source_param_ip_address;
wire [15:0] tx_cdc_source_param_length;
wire tx_converter_sink_sink_valid;
wire tx_converter_sink_sink_ready;
wire tx_converter_sink_sink_first;
wire tx_converter_sink_sink_last;
wire [7:0] tx_converter_sink_sink_payload_data;
wire tx_converter_sink_sink_payload_be;
wire tx_converter_sink_sink_payload_error;
wire [15:0] tx_converter_sink_sink_param_src_port;
wire [15:0] tx_converter_sink_sink_param_dst_port;
wire [31:0] tx_converter_sink_sink_param_ip_address;
wire [15:0] tx_converter_sink_sink_param_length;
wire tx_converter_source_source_valid;
wire tx_converter_source_source_ready;
wire tx_converter_source_source_first;
wire tx_converter_source_source_last;
wire [7:0] tx_converter_source_source_payload_data;
wire tx_converter_source_source_payload_be;
wire tx_converter_source_source_payload_error;
wire [15:0] tx_converter_source_source_param_src_port;
wire [15:0] tx_converter_source_source_param_dst_port;
wire [31:0] tx_converter_source_source_param_ip_address;
wire [15:0] tx_converter_source_source_param_length;
wire tx_converter_converter_sink_valid;
wire tx_converter_converter_sink_ready;
wire tx_converter_converter_sink_first;
wire tx_converter_converter_sink_last;
wire [9:0] tx_converter_converter_sink_payload_data;
wire tx_converter_converter_source_valid;
wire tx_converter_converter_source_ready;
wire tx_converter_converter_source_first;
wire tx_converter_converter_source_last;
wire [9:0] tx_converter_converter_source_payload_data;
wire tx_converter_converter_source_payload_valid_token_count;
wire rx_converter_sink_sink_valid;
wire rx_converter_sink_sink_ready;
wire rx_converter_sink_sink_first;
wire rx_converter_sink_sink_last;
wire [7:0] rx_converter_sink_sink_payload_data;
wire rx_converter_sink_sink_payload_be;
wire rx_converter_sink_sink_payload_error;
wire [15:0] rx_converter_sink_sink_param_src_port;
wire [15:0] rx_converter_sink_sink_param_dst_port;
wire [31:0] rx_converter_sink_sink_param_ip_address;
wire [15:0] rx_converter_sink_sink_param_length;
wire rx_converter_source_source_valid;
wire rx_converter_source_source_ready;
wire rx_converter_source_source_first;
wire rx_converter_source_source_last;
wire [7:0] rx_converter_source_source_payload_data;
wire rx_converter_source_source_payload_be;
wire rx_converter_source_source_payload_error;
wire [15:0] rx_converter_source_source_param_src_port;
wire [15:0] rx_converter_source_source_param_dst_port;
wire [31:0] rx_converter_source_source_param_ip_address;
wire [15:0] rx_converter_source_source_param_length;
wire rx_converter_converter_sink_valid;
wire rx_converter_converter_sink_ready;
wire rx_converter_converter_sink_first;
wire rx_converter_converter_sink_last;
wire [9:0] rx_converter_converter_sink_payload_data;
wire rx_converter_converter_source_valid;
wire rx_converter_converter_source_ready;
wire rx_converter_converter_source_first;
wire rx_converter_converter_source_last;
wire [9:0] rx_converter_converter_source_payload_data;
wire rx_converter_converter_source_payload_valid_token_count;
wire rx_cdc_sink_valid;
wire rx_cdc_sink_ready;
wire rx_cdc_sink_first;
wire rx_cdc_sink_last;
wire [7:0] rx_cdc_sink_payload_data;
wire rx_cdc_sink_payload_be;
wire rx_cdc_sink_payload_error;
wire [15:0] rx_cdc_sink_param_src_port;
wire [15:0] rx_cdc_sink_param_dst_port;
wire [31:0] rx_cdc_sink_param_ip_address;
wire [15:0] rx_cdc_sink_param_length;
wire rx_cdc_source_valid;
wire rx_cdc_source_ready;
wire rx_cdc_source_first;
wire rx_cdc_source_last;
wire [7:0] rx_cdc_source_payload_data;
wire rx_cdc_source_payload_be;
wire rx_cdc_source_payload_error;
wire [15:0] rx_cdc_source_param_src_port;
wire [15:0] rx_cdc_source_param_dst_port;
wire [31:0] rx_cdc_source_param_ip_address;
wire [15:0] rx_cdc_source_param_length;
wire echo_buffer_sink_valid;
wire echo_buffer_sink_ready;
wire echo_buffer_sink_first;
wire echo_buffer_sink_last;
wire [7:0] echo_buffer_sink_payload_data;
wire echo_buffer_sink_payload_be;
wire echo_buffer_sink_payload_error;
wire [15:0] echo_buffer_sink_param_src_port;
wire [15:0] echo_buffer_sink_param_dst_port;
wire [31:0] echo_buffer_sink_param_ip_address;
wire [15:0] echo_buffer_sink_param_length;
wire echo_buffer_source_valid;
wire echo_buffer_source_ready;
wire echo_buffer_source_first;
wire echo_buffer_source_last;
wire [7:0] echo_buffer_source_payload_data;
wire echo_buffer_source_payload_be;
wire echo_buffer_source_payload_error;
wire [15:0] echo_buffer_source_param_src_port;
wire [15:0] echo_buffer_source_param_dst_port;
wire [31:0] echo_buffer_source_param_ip_address;
wire [15:0] echo_buffer_source_param_length;
wire echo_buffer_re;
reg echo_buffer_readable = 1'd0;
wire echo_buffer_syncfifo_we;
wire echo_buffer_syncfifo_writable;
wire echo_buffer_syncfifo_re;
wire echo_buffer_syncfifo_readable;
wire [91:0] echo_buffer_syncfifo_din;
wire [91:0] echo_buffer_syncfifo_dout;
reg [8:0] echo_buffer_level0 = 9'd0;
reg echo_buffer_replace = 1'd0;
reg [7:0] echo_buffer_produce = 8'd0;
reg [7:0] echo_buffer_consume = 8'd0;
reg [7:0] echo_buffer_wrport_adr;
wire echo_buffer_wrport_we;
wire [91:0] echo_buffer_wrport_dat_w;
wire echo_buffer_do_read;
wire [7:0] echo_buffer_rdport_adr;
wire [91:0] echo_buffer_rdport_dat_r;
wire echo_buffer_rdport_re;
wire [8:0] echo_buffer_level1;
wire [7:0] echo_buffer_fifo_in_payload_data;
wire echo_buffer_fifo_in_payload_be;
wire echo_buffer_fifo_in_payload_error;
wire [15:0] echo_buffer_fifo_in_param_src_port;
wire [15:0] echo_buffer_fifo_in_param_dst_port;
wire [31:0] echo_buffer_fifo_in_param_ip_address;
wire [15:0] echo_buffer_fifo_in_param_length;
wire echo_buffer_fifo_in_first;
wire echo_buffer_fifo_in_last;
wire [7:0] echo_buffer_fifo_out_payload_data;
wire echo_buffer_fifo_out_payload_be;
wire echo_buffer_fifo_out_payload_error;
wire [15:0] echo_buffer_fifo_out_param_src_port;
wire [15:0] echo_buffer_fifo_out_param_dst_port;
wire [31:0] echo_buffer_fifo_out_param_ip_address;
wire [15:0] echo_buffer_fifo_out_param_length;
wire echo_buffer_fifo_out_first;
wire echo_buffer_fifo_out_last;
wire dbg_led_shim_last1;
reg [7:0] hold0 = 8'd0;
wire dbg_led_cdc_last1;
reg [7:0] hold1 = 8'd0;
wire dbg_led_mac_last1;
reg [7:0] hold2 = 8'd0;
wire dbg_led_arp_req1;
reg [7:0] hold3 = 8'd0;
wire dbg_led_arp_rep1;
reg [7:0] hold4 = 8'd0;
wire dbg_led_tx_last1;
reg [7:0] hold5 = 8'd0;
reg [7:0] value0 = 8'd0;
reg [7:0] value1 = 8'd0;
reg [7:0] value2 = 8'd0;
reg [7:0] value3 = 8'd0;
reg [7:0] value4 = 8'd0;
reg [7:0] value5 = 8'd0;
reg [7:0] value6 = 8'd0;
reg [7:0] value7 = 8'd0;
wire [7:0] uarttx_data;
wire uarttx_strobe;
wire uarttx_busy;
reg [1:0] uarttx_state = 2'd0;
reg [9:0] uarttx_cnt = 10'd0;
reg [7:0] uarttx_shreg = 8'd0;
reg [2:0] uarttx_bitidx = 3'd0;
reg uarttx_pend_bit = 1'd0;
reg [7:0] uarttx_pend_data = 8'd0;
reg uarttx = 1'd0;
reg [22:0] timer = 23'd8388607;
reg fire = 1'd0;
reg [3:0] idx = 4'd0;
reg dumping = 1'd0;
reg liteethmac_txdatapath_liteethmacpaddinginserter_state = 1'd0;
reg liteethmac_txdatapath_liteethmacpaddinginserter_next_state;
reg [15:0] mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value;
reg mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value_ce;
reg [1:0] liteethmac_txdatapath_bufferizeendpoints_state = 2'd0;
reg [1:0] liteethmac_txdatapath_bufferizeendpoints_next_state;
reg [31:0] mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value0;
reg mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value_ce0;
reg mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value1;
reg mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value_ce1;
reg [1:0] liteethmac_txdatapath_liteethmacpreambleinserter_state = 2'd0;
reg [1:0] liteethmac_txdatapath_liteethmacpreambleinserter_next_state;
reg [2:0] mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value;
reg mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value_ce;
reg liteethmac_txdatapath_resetinserter_state = 1'd0;
reg liteethmac_txdatapath_resetinserter_next_state;
reg [3:0] mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value;
reg mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value_ce;
reg liteethmac_rxdatapath_liteethmacpreamblechecker_state = 1'd0;
reg liteethmac_rxdatapath_liteethmacpreamblechecker_next_state;
reg [1:0] liteethmac_rxdatapath_bufferizeendpoints_state = 2'd0;
reg [1:0] liteethmac_rxdatapath_bufferizeendpoints_next_state;
reg mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value0;
reg mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value_ce0;
reg mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value1;
reg mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value_ce1;
reg [1:0] liteethmac_request;
reg liteethmac_grant = 1'd0;
reg liteethmac_status0_first = 1'd1;
wire liteethmac_status0_last;
wire liteethmac_status0_ongoing0;
reg liteethmac_status0_ongoing1 = 1'd0;
reg liteethmac_status1_first = 1'd1;
wire liteethmac_status1_last;
wire liteethmac_status1_ongoing0;
reg liteethmac_status1_ongoing1 = 1'd0;
reg [1:0] liteethmac_sel0;
reg liteethmac_first = 1'd1;
wire liteethmac_last;
wire liteethmac_ongoing0;
reg liteethmac_ongoing1 = 1'd0;
reg [1:0] liteethmac_sel1;
reg [1:0] liteethmac_sel_ongoing = 2'd0;
reg liteethmac_sel_locked = 1'd0;
reg liteethmac_liteethmacpacketizer_state = 1'd1;
reg liteethmac_liteethmacpacketizer_next_state;
reg [3:0] mac_packetizer_count_liteethmacpacketizer_next_value;
reg mac_packetizer_count_liteethmacpacketizer_next_value_ce;
reg [1:0] liteethmac_liteethmacdepacketizer_state = 2'd0;
reg [1:0] liteethmac_liteethmacdepacketizer_next_state;
reg [3:0] mac_depacketizer_count_liteethmacdepacketizer_next_value;
reg mac_depacketizer_count_liteethmacdepacketizer_next_value_ce;
reg mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value0;
reg mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value_ce0;
reg mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value1;
reg mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value_ce1;
reg mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value;
reg mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value_ce;
reg liteetharptx_liteetharppacketizer_state = 1'd1;
reg liteetharptx_liteetharppacketizer_next_state;
reg [4:0] arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value;
reg arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value_ce;
reg liteetharptx_fsm_state = 1'd0;
reg liteetharptx_fsm_next_state;
reg [4:0] arp_tx_counter_liteetharp_liteetharptx_fsm_next_value;
reg arp_tx_counter_liteetharp_liteetharptx_fsm_next_value_ce;
reg [1:0] liteetharprx_liteetharpdepacketizer_state = 2'd0;
reg [1:0] liteetharprx_liteetharpdepacketizer_next_state;
reg [4:0] arp_rx_depacketizer_count_liteetharp_next_value;
reg arp_rx_depacketizer_count_liteetharp_next_value_ce;
reg arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value0;
reg arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value_ce0;
reg arp_rx_depacketizer_send_dummy_liteetharp_t_next_value1;
reg arp_rx_depacketizer_send_dummy_liteetharp_t_next_value_ce1;
reg arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value;
reg arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value_ce;
reg [1:0] liteetharprx_fsm_state = 2'd0;
reg [1:0] liteetharprx_fsm_next_state;
reg [2:0] liteetharptable_liteetharpcache_state = 3'd0;
reg [2:0] liteetharptable_liteetharpcache_next_state;
reg arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0;
reg arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0;
reg arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value1;
reg arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce1;
reg arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value2;
reg arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value_ce2;
reg [2:0] liteetharptable_fsm_state = 3'd0;
reg [2:0] liteetharptable_fsm_next_state;
reg arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value0;
reg arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value_ce0;
reg [47:0] arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value1;
reg arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value_ce1;
reg arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value2;
reg arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value_ce2;
reg [3:0] arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value3;
reg arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value_ce3;
reg [31:0] arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value4;
reg arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value_ce4;
reg liteethip_liteethiptx_liteethipv4packetizer_state = 1'd1;
reg liteethip_liteethiptx_liteethipv4packetizer_next_state;
reg [4:0] ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value;
reg ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value_ce;
reg [2:0] liteethip_liteethiptx_fsm_state = 3'd2;
reg [2:0] liteethip_liteethiptx_fsm_next_state;
reg [47:0] ip_tx_target_mac_liteethip_fsm_next_value;
reg ip_tx_target_mac_liteethip_fsm_next_value_ce;
reg [1:0] liteethip_liteethiprx_liteethipv4depacketizer_state = 2'd0;
reg [1:0] liteethip_liteethiprx_liteethipv4depacketizer_next_state;
reg [4:0] ip_rx_depacketizer_count_liteethip_next_value;
reg ip_rx_depacketizer_count_liteethip_next_value_ce;
reg ip_rx_depacketizer_data_copy_first_liteethip_t_next_value0;
reg ip_rx_depacketizer_data_copy_first_liteethip_t_next_value_ce0;
reg ip_rx_depacketizer_send_dummy_liteethip_t_next_value1;
reg ip_rx_depacketizer_send_dummy_liteethip_t_next_value_ce1;
reg ip_rx_depacketizer_skip_cnt_liteethip_f_next_value;
reg ip_rx_depacketizer_skip_cnt_liteethip_f_next_value_ce;
reg [1:0] liteethip_liteethiprx_fsm_state = 2'd1;
reg [1:0] liteethip_liteethiprx_fsm_next_state;
reg [1:0] liteethip_request;
reg liteethip_grant = 1'd0;
reg liteethip_status0_first = 1'd1;
wire liteethip_status0_last;
wire liteethip_status0_ongoing0;
reg liteethip_status0_ongoing1 = 1'd0;
reg liteethip_status1_first = 1'd1;
wire liteethip_status1_last;
wire liteethip_status1_ongoing0;
reg liteethip_status1_ongoing1 = 1'd0;
reg [1:0] liteethip_sel0;
reg liteethip_first = 1'd1;
wire liteethip_last;
wire liteethip_ongoing0;
reg liteethip_ongoing1 = 1'd0;
reg [1:0] liteethip_sel1;
reg [1:0] liteethip_sel_ongoing = 2'd0;
reg liteethip_sel_locked = 1'd0;
reg liteethicmp_state = 1'd1;
reg liteethicmp_next_state;
reg [2:0] icmp_tx_packetizer_count_liteethicmp_next_value;
reg icmp_tx_packetizer_count_liteethicmp_next_value_ce;
reg [1:0] liteethicmp_liteethicmpdepacketizer_state = 2'd0;
reg [1:0] liteethicmp_liteethicmpdepacketizer_next_state;
reg [2:0] icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value0;
reg icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value_ce0;
reg icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value1;
reg icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value_ce1;
reg icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value2;
reg icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value_ce2;
reg [1:0] liteethicmp_fsm_state = 2'd2;
reg [1:0] liteethicmp_fsm_next_state;
reg [16:0] icmp_rx_count_liteethicmp_fsm_next_value;
reg icmp_rx_count_liteethicmp_fsm_next_value_ce;
reg liteethudp_state = 1'd1;
reg liteethudp_next_state;
reg [2:0] tx_packetizer_count_liteethudp_next_value;
reg tx_packetizer_count_liteethudp_next_value_ce;
reg [1:0] liteethudp_liteethudpdepacketizer_state = 2'd0;
reg [1:0] liteethudp_liteethudpdepacketizer_next_state;
reg [2:0] rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value;
reg rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value_ce;
reg rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value0;
reg rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value_ce0;
reg rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value1;
reg rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value_ce1;
reg rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value;
reg rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value_ce;
reg [1:0] liteethudp_fsm_state = 2'd2;
reg [1:0] liteethudp_fsm_next_state;
reg [16:0] rx_count_liteethudp_fsm_next_value;
reg rx_count_liteethudp_fsm_next_value_ce;
reg liteethudp_sel0;
reg liteethudp_first = 1'd1;
wire liteethudp_last;
wire liteethudp_ongoing0;
reg liteethudp_ongoing1 = 1'd0;
reg liteethudp_sel1;
reg liteethudp_sel_ongoing = 1'd0;
reg liteethudp_sel_locked = 1'd0;
wire [39:0] complexslicelowerer_slice_proxy;
wire [15:0] rhs_slice_proxy0;
wire [15:0] rhs_slice_proxy1;
wire [47:0] rhs_slice_proxy2;
wire [47:0] rhs_slice_proxy3;
wire [47:0] rhs_slice_proxy4;
wire [47:0] rhs_slice_proxy5;
wire [47:0] rhs_slice_proxy6;
wire [47:0] rhs_slice_proxy7;
wire [47:0] rhs_slice_proxy8;
wire [47:0] rhs_slice_proxy9;
wire [47:0] rhs_slice_proxy10;
wire [47:0] rhs_slice_proxy11;
wire [47:0] rhs_slice_proxy12;
wire [47:0] rhs_slice_proxy13;
wire [7:0] rhs_slice_proxy14;
wire [15:0] rhs_slice_proxy15;
wire [15:0] rhs_slice_proxy16;
wire [15:0] rhs_slice_proxy17;
wire [15:0] rhs_slice_proxy18;
wire [15:0] rhs_slice_proxy19;
wire [15:0] rhs_slice_proxy20;
wire [7:0] rhs_slice_proxy21;
wire [31:0] rhs_slice_proxy22;
wire [31:0] rhs_slice_proxy23;
wire [31:0] rhs_slice_proxy24;
wire [31:0] rhs_slice_proxy25;
wire [47:0] rhs_slice_proxy26;
wire [47:0] rhs_slice_proxy27;
wire [47:0] rhs_slice_proxy28;
wire [47:0] rhs_slice_proxy29;
wire [47:0] rhs_slice_proxy30;
wire [47:0] rhs_slice_proxy31;
wire [31:0] rhs_slice_proxy32;
wire [31:0] rhs_slice_proxy33;
wire [31:0] rhs_slice_proxy34;
wire [31:0] rhs_slice_proxy35;
wire [47:0] rhs_slice_proxy36;
wire [47:0] rhs_slice_proxy37;
wire [47:0] rhs_slice_proxy38;
wire [47:0] rhs_slice_proxy39;
wire [47:0] rhs_slice_proxy40;
wire [47:0] rhs_slice_proxy41;
wire [15:0] rhs_slice_proxy42;
wire [15:0] rhs_slice_proxy43;
wire [15:0] rhs_slice_proxy44;
wire [15:0] rhs_slice_proxy45;
wire [15:0] rhs_slice_proxy46;
wire [15:0] rhs_slice_proxy47;
wire [3:0] rhs_slice_proxy48;
wire [7:0] rhs_slice_proxy49;
wire [31:0] rhs_slice_proxy50;
wire [31:0] rhs_slice_proxy51;
wire [31:0] rhs_slice_proxy52;
wire [31:0] rhs_slice_proxy53;
wire [31:0] rhs_slice_proxy54;
wire [31:0] rhs_slice_proxy55;
wire [31:0] rhs_slice_proxy56;
wire [31:0] rhs_slice_proxy57;
wire [15:0] rhs_slice_proxy58;
wire [15:0] rhs_slice_proxy59;
wire [7:0] rhs_slice_proxy60;
wire [3:0] rhs_slice_proxy61;
wire [15:0] rhs_slice_proxy62;
wire [15:0] rhs_slice_proxy63;
wire [7:0] rhs_slice_proxy64;
wire [7:0] rhs_slice_proxy65;
wire [31:0] rhs_slice_proxy66;
wire [31:0] rhs_slice_proxy67;
wire [31:0] rhs_slice_proxy68;
wire [31:0] rhs_slice_proxy69;
wire [15:0] rhs_slice_proxy70;
wire [15:0] rhs_slice_proxy71;
wire [15:0] rhs_slice_proxy72;
wire [15:0] rhs_slice_proxy73;
wire [15:0] rhs_slice_proxy74;
wire [15:0] rhs_slice_proxy75;
wire [15:0] rhs_slice_proxy76;
wire [15:0] rhs_slice_proxy77;
reg [7:0] self;
(* no_retiming = "true" *) reg multiregimpl00 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl01 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl10 = 1'd0;
(* no_retiming = "true" *) reg multiregimpl11 = 1'd0;

// synthesis translate_off
reg dummy_s;
initial dummy_s <= 1'd0;
// synthesis translate_on

assign echo_buffer_sink_valid = user_port_source_valid;
assign user_port_source_ready = echo_buffer_sink_ready;
assign echo_buffer_sink_first = user_port_source_first;
assign echo_buffer_sink_last = user_port_source_last;
assign echo_buffer_sink_payload_data = user_port_source_payload_data;
assign echo_buffer_sink_payload_be = user_port_source_payload_be;
assign echo_buffer_sink_payload_error = user_port_source_payload_error;
assign echo_buffer_sink_param_src_port = user_port_source_param_src_port;
assign echo_buffer_sink_param_dst_port = user_port_source_param_dst_port;
assign echo_buffer_sink_param_ip_address = user_port_source_param_ip_address;
assign echo_buffer_sink_param_length = user_port_source_param_length;
assign user_port_sink_valid = echo_buffer_source_valid;
assign echo_buffer_source_ready = user_port_sink_ready;
assign user_port_sink_first = echo_buffer_source_first;
assign user_port_sink_last = echo_buffer_source_last;
assign user_port_sink_payload_data = echo_buffer_source_payload_data;
assign user_port_sink_payload_be = echo_buffer_source_payload_be;
assign user_port_sink_payload_error = echo_buffer_source_payload_error;
assign user_port_sink_param_ip_address = echo_buffer_source_param_ip_address;
assign user_port_sink_param_length = echo_buffer_source_param_length;
assign user_port_sink_param_src_port = 11'd1234;
assign user_port_sink_param_dst_port = echo_buffer_source_param_src_port;
assign dbg_led_shim_last1 = (hold0 != 1'd0);
assign dbg_led_cdc_last1 = (hold1 != 1'd0);
assign dbg_led_mac_last1 = (hold2 != 1'd0);
assign dbg_led_arp_req1 = (hold3 != 1'd0);
assign dbg_led_arp_rep1 = (hold4 != 1'd0);
assign dbg_led_tx_last1 = (hold5 != 1'd0);
assign dbg_crs_dv = rmii_crs_dv;
assign dbg_rx_mac = phy_source_valid;
assign dbg_tx_mac = phy_sink_valid;
assign dbg_rx_udp = user_port_sink_valid;
assign dbg_mac_in = mac_depacketizer_sink_valid;
assign dbg_mac_out = mac_depacketizer_source_valid;
assign dbg_arp_rx = (mac_depacketizer_sink_valid & mac_depacketizer_sink_last);
assign dbg_arp_req = (arp_table_sink_valid & arp_table_sink_payload_request);
assign dbg_arp_rep = arp_table_source_valid;
assign dbg_pre_err = 1'd0;
assign dbg_crc_err = 1'd0;
assign dbg_led_shim_last = dbg_led_shim_last1;
assign dbg_led_cdc_last = dbg_led_cdc_last1;
assign dbg_led_mac_last = dbg_led_mac_last1;
assign dbg_led_arp_req = dbg_led_arp_req1;
assign dbg_led_arp_rep = dbg_led_arp_rep1;
assign dbg_led_tx_last = dbg_led_tx_last1;
assign dbg_led_rx_err = 1'd0;
assign uarttx_strobe = (fire | (dumping & (~uarttx_busy)));
assign uarttx_data = (fire ? 8'd170 : self);
assign phy_rmiitxshim_sink_valid = phy_sink_valid;
assign phy_sink_ready = phy_rmiitxshim_sink_ready;
assign phy_rmiitxshim_sink_first = phy_sink_first;
assign phy_rmiitxshim_sink_last = phy_sink_last;
assign phy_rmiitxshim_sink_payload_data = phy_sink_payload_data;
assign phy_rmiitxshim_sink_payload_be = phy_sink_payload_be;
assign phy_rmiitxshim_sink_payload_error = phy_sink_payload_error;
assign phy_source_valid = phy_rmiirxshim_source_valid;
assign phy_rmiirxshim_source_ready = phy_source_ready;
assign phy_source_first = phy_rmiirxshim_source_first;
assign phy_source_last = phy_rmiirxshim_source_last;
assign phy_source_payload_data = phy_rmiirxshim_source_payload_data;
assign phy_source_payload_be = phy_rmiirxshim_source_payload_be;
assign phy_source_payload_error = phy_rmiirxshim_source_payload_error;
assign rmii_txd = phy_rmiitxshim_shift[1:0];
assign rmii_tx_en = (phy_rmiitxshim_count != 1'd0);
assign phy_rmiitxshim_sink_ready = (phy_rmiitxshim_count <= 1'd1);
assign phy_rmiirxshim_source_payload_be = 1'd1;
assign phy_rmiirxshim_crs_dv = rmii_crs_dv;
assign phy_rmiirxshim_rx_data = rmii_rxd;
assign phy_rmiirxshim_crs_first = (phy_rmiirxshim_crs_dv & (phy_rmiirxshim_rx_data != 1'd0));
assign phy_rmiirxshim_crs_last = ((~phy_rmiirxshim_crs_dv) & (~phy_rmiirxshim_crs_dv_d));
assign phy_rmiirxshim_cap = (phy_rmiirxshim_crs_first | phy_rmiirxshim_crs_run);
assign phy_rmiirxshim_sfd_hit = (((phy_rmiirxshim_e1 & (phy_rmiirxshim_mode == 1'd0)) & (phy_rmiirxshim_sreg[5:0] == 5'd21)) & (phy_rmiirxshim_d1 == 2'd3));
assign phy_rmiirxshim_pre_byte = ((phy_rmiirxshim_pre_cnt == 3'd7) ? 8'd213 : 7'd85);
assign phy_rmiirxshim_source_valid = ((phy_rmiirxshim_mode == 1'd1) | ((phy_rmiirxshim_mode == 2'd2) & phy_rmiirxshim_f0_valid));
assign phy_rmiirxshim_source_payload_data = ((phy_rmiirxshim_mode == 1'd1) ? phy_rmiirxshim_pre_byte : phy_rmiirxshim_f0_data);
assign phy_rmiirxshim_source_last = ((phy_rmiirxshim_mode == 2'd2) & phy_rmiirxshim_f0_last);
assign phy_rmiirxshim_consume = (phy_rmiirxshim_source_valid & phy_rmiirxshim_source_ready);
assign phy_rmiirxshim_new_byte = {phy_rmiirxshim_d1, phy_rmiirxshim_shift[7:2]};
assign phy_rmiirxshim_pushing = (((phy_rmiirxshim_mode != 1'd0) & phy_rmiirxshim_e1) & (phy_rmiirxshim_bits == 2'd3));
assign mac_packetizer_sink_valid = mac_crossbar_source_valid;
assign mac_crossbar_source_ready = mac_packetizer_sink_ready;
assign mac_packetizer_sink_first = mac_crossbar_source_first;
assign mac_packetizer_sink_last = mac_crossbar_source_last;
assign mac_packetizer_sink_payload_ethernet_type = mac_crossbar_source_payload_ethernet_type;
assign mac_packetizer_sink_payload_sender_mac = mac_crossbar_source_payload_sender_mac;
assign mac_packetizer_sink_payload_target_mac = mac_crossbar_source_payload_target_mac;
assign mac_packetizer_sink_payload_data = mac_crossbar_source_payload_data;
assign mac_packetizer_sink_payload_be = mac_crossbar_source_payload_be;
assign mac_packetizer_sink_payload_error = mac_crossbar_source_payload_error;
assign mac_core_sink_valid = mac_packetizer_source_valid;
assign mac_packetizer_source_ready = mac_core_sink_ready;
assign mac_core_sink_first = mac_packetizer_source_first;
assign mac_core_sink_last = mac_packetizer_source_last;
assign mac_core_sink_payload_data = mac_packetizer_source_payload_data;
assign mac_core_sink_payload_be = mac_packetizer_source_payload_be;
assign mac_core_sink_payload_error = mac_packetizer_source_payload_error;
assign mac_depacketizer_sink_valid = mac_core_source_valid;
assign mac_core_source_ready = mac_depacketizer_sink_ready;
assign mac_depacketizer_sink_first = mac_core_source_first;
assign mac_depacketizer_sink_last = mac_core_source_last;
assign mac_depacketizer_sink_payload_data = mac_core_source_payload_data;
assign mac_depacketizer_sink_payload_be = mac_core_source_payload_be;
assign mac_depacketizer_sink_payload_error = mac_core_source_payload_error;
assign mac_crossbar_sink_valid = mac_depacketizer_source_valid;
assign mac_depacketizer_source_ready = mac_crossbar_sink_ready;
assign mac_crossbar_sink_first = mac_depacketizer_source_first;
assign mac_crossbar_sink_last = mac_depacketizer_source_last;
assign mac_crossbar_sink_payload_ethernet_type = mac_depacketizer_source_payload_ethernet_type;
assign mac_crossbar_sink_payload_sender_mac = mac_depacketizer_source_payload_sender_mac;
assign mac_crossbar_sink_payload_target_mac = mac_depacketizer_source_payload_target_mac;
assign mac_crossbar_sink_payload_data = mac_depacketizer_source_payload_data;
assign mac_crossbar_sink_payload_be = mac_depacketizer_source_payload_be;
assign mac_crossbar_sink_payload_error = mac_depacketizer_source_payload_error;
assign mac_core_cdc_rst = mac_core_txdatapath_ext_rst_to;
assign mac_core_tx_gap_reset = mac_core_cdc_rst;
assign mac_core_tx_padding_counter_done = (mac_core_tx_padding_counter >= 6'd59);

// synthesis translate_off
reg dummy_d;
// synthesis translate_on
always @(*) begin
	mac_core_tx_padding_sink_ready <= 1'd0;
	mac_core_tx_padding_source_valid <= 1'd0;
	mac_core_tx_padding_source_first <= 1'd0;
	mac_core_tx_padding_source_last <= 1'd0;
	mac_core_tx_padding_source_payload_data <= 8'd0;
	mac_core_tx_padding_source_payload_be <= 1'd0;
	mac_core_tx_padding_source_payload_error <= 1'd0;
	liteethmac_txdatapath_liteethmacpaddinginserter_next_state <= 1'd0;
	mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value <= 16'd0;
	mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value_ce <= 1'd0;
	liteethmac_txdatapath_liteethmacpaddinginserter_next_state <= liteethmac_txdatapath_liteethmacpaddinginserter_state;
	case (liteethmac_txdatapath_liteethmacpaddinginserter_state)
		1'd1: begin
			mac_core_tx_padding_source_valid <= 1'd1;
			mac_core_tx_padding_source_payload_be <= 1'd1;
			if (mac_core_tx_padding_counter_done) begin
				mac_core_tx_padding_source_payload_be <= 1'd1;
				mac_core_tx_padding_source_last <= 1'd1;
			end
			mac_core_tx_padding_source_payload_data <= 1'd0;
			if ((mac_core_tx_padding_source_valid & mac_core_tx_padding_source_ready)) begin
				mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value <= (mac_core_tx_padding_counter + 1'd1);
				mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value_ce <= 1'd1;
				if (mac_core_tx_padding_counter_done) begin
					mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value <= 1'd0;
					mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value_ce <= 1'd1;
					liteethmac_txdatapath_liteethmacpaddinginserter_next_state <= 1'd0;
				end
			end
		end
		default: begin
			mac_core_tx_padding_source_valid <= mac_core_tx_padding_sink_valid;
			mac_core_tx_padding_sink_ready <= mac_core_tx_padding_source_ready;
			mac_core_tx_padding_source_first <= mac_core_tx_padding_sink_first;
			mac_core_tx_padding_source_last <= mac_core_tx_padding_sink_last;
			mac_core_tx_padding_source_payload_data <= mac_core_tx_padding_sink_payload_data;
			mac_core_tx_padding_source_payload_be <= mac_core_tx_padding_sink_payload_be;
			mac_core_tx_padding_source_payload_error <= mac_core_tx_padding_sink_payload_error;
			if (mac_core_tx_padding_sink_last) begin
				if ((~mac_core_tx_padding_sink_payload_be)) begin
					mac_core_tx_padding_source_payload_data[7:0] <= 1'd0;
				end
				if ((~mac_core_tx_padding_counter_done)) begin
					mac_core_tx_padding_source_last <= 1'd0;
					mac_core_tx_padding_source_payload_be <= 1'd1;
				end else begin
					if (((mac_core_tx_padding_counter == 6'd59) & (mac_core_tx_padding_sink_payload_be < 1'd1))) begin
						mac_core_tx_padding_source_payload_be <= 1'd1;
					end
				end
			end
			if ((mac_core_tx_padding_source_valid & mac_core_tx_padding_source_ready)) begin
				mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value <= (mac_core_tx_padding_counter + 1'd1);
				mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value_ce <= 1'd1;
				if (mac_core_tx_padding_sink_last) begin
					if ((~mac_core_tx_padding_counter_done)) begin
						liteethmac_txdatapath_liteethmacpaddinginserter_next_state <= 1'd1;
					end else begin
						mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value <= 1'd0;
						mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value_ce <= 1'd1;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d <= dummy_s;
// synthesis translate_on
end
assign mac_core_tx_crc_data0 = mac_core_tx_crc_sink_payload_data;
assign mac_core_tx_crc_be1 = mac_core_tx_crc_sink_payload_be;
assign mac_core_tx_crc_cnt_done = (mac_core_tx_crc_cnt == 1'd0);
assign mac_core_tx_crc_sink_valid = mac_core_tx_crc_source_source_valid;
assign mac_core_tx_crc_source_source_ready = mac_core_tx_crc_sink_ready;
assign mac_core_tx_crc_sink_first = mac_core_tx_crc_source_source_first;
assign mac_core_tx_crc_sink_last = mac_core_tx_crc_source_source_last;
assign mac_core_tx_crc_sink_payload_data = mac_core_tx_crc_source_source_payload_data;
assign mac_core_tx_crc_sink_payload_be = mac_core_tx_crc_source_source_payload_be;
assign mac_core_tx_crc_sink_payload_error = mac_core_tx_crc_source_source_payload_error;
assign mac_core_tx_crc_data1 = mac_core_tx_crc_data0;
assign mac_core_tx_crc_crc_prev = mac_core_tx_crc_reg;

// synthesis translate_off
reg dummy_d_1;
// synthesis translate_on
always @(*) begin
	mac_core_tx_crc_value <= 32'd0;
	mac_core_tx_crc_error <= 1'd0;
	if (mac_core_tx_crc_be1) begin
		mac_core_tx_crc_value <= ({mac_core_tx_crc_crc_next[0], mac_core_tx_crc_crc_next[1], mac_core_tx_crc_crc_next[2], mac_core_tx_crc_crc_next[3], mac_core_tx_crc_crc_next[4], mac_core_tx_crc_crc_next[5], mac_core_tx_crc_crc_next[6], mac_core_tx_crc_crc_next[7], mac_core_tx_crc_crc_next[8], mac_core_tx_crc_crc_next[9], mac_core_tx_crc_crc_next[10], mac_core_tx_crc_crc_next[11], mac_core_tx_crc_crc_next[12], mac_core_tx_crc_crc_next[13], mac_core_tx_crc_crc_next[14], mac_core_tx_crc_crc_next[15], mac_core_tx_crc_crc_next[16], mac_core_tx_crc_crc_next[17], mac_core_tx_crc_crc_next[18], mac_core_tx_crc_crc_next[19], mac_core_tx_crc_crc_next[20], mac_core_tx_crc_crc_next[21], mac_core_tx_crc_crc_next[22], mac_core_tx_crc_crc_next[23], mac_core_tx_crc_crc_next[24], mac_core_tx_crc_crc_next[25], mac_core_tx_crc_crc_next[26], mac_core_tx_crc_crc_next[27], mac_core_tx_crc_crc_next[28], mac_core_tx_crc_crc_next[29], mac_core_tx_crc_crc_next[30], mac_core_tx_crc_crc_next[31]} ^ 32'd4294967295);
		mac_core_tx_crc_error <= (mac_core_tx_crc_crc_next != 32'd3338984827);
	end
// synthesis translate_off
	dummy_d_1 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_2;
// synthesis translate_on
always @(*) begin
	mac_core_tx_crc_crc_next <= 32'd0;
	mac_core_tx_crc_crc_next[0] <= (((mac_core_tx_crc_crc_prev[24] ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[1] <= (((((((mac_core_tx_crc_crc_prev[25] ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[2] <= (((((((((mac_core_tx_crc_crc_prev[26] ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[3] <= (((((((mac_core_tx_crc_crc_prev[27] ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]);
	mac_core_tx_crc_crc_next[4] <= (((((((((mac_core_tx_crc_crc_prev[28] ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[5] <= (((((((((((((mac_core_tx_crc_crc_prev[29] ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[6] <= (((((((((((mac_core_tx_crc_crc_prev[30] ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]);
	mac_core_tx_crc_crc_next[7] <= (((((((((mac_core_tx_crc_crc_prev[31] ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[8] <= ((((((((mac_core_tx_crc_crc_prev[0] ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_data1[6]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[9] <= ((((((((mac_core_tx_crc_crc_prev[1] ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_data1[6]);
	mac_core_tx_crc_crc_next[10] <= ((((((((mac_core_tx_crc_crc_prev[2] ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[11] <= ((((((((mac_core_tx_crc_crc_prev[3] ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_data1[6]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[12] <= ((((((((((((mac_core_tx_crc_crc_prev[4] ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_data1[6]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[13] <= ((((((((((((mac_core_tx_crc_crc_prev[5] ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]);
	mac_core_tx_crc_crc_next[14] <= ((((((((((mac_core_tx_crc_crc_prev[6] ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]);
	mac_core_tx_crc_crc_next[15] <= ((((((((mac_core_tx_crc_crc_prev[7] ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]);
	mac_core_tx_crc_crc_next[16] <= ((((((mac_core_tx_crc_crc_prev[8] ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[17] <= ((((((mac_core_tx_crc_crc_prev[9] ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_data1[6]);
	mac_core_tx_crc_crc_next[18] <= ((((((mac_core_tx_crc_crc_prev[10] ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]);
	mac_core_tx_crc_crc_next[19] <= ((((mac_core_tx_crc_crc_prev[11] ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]);
	mac_core_tx_crc_crc_next[20] <= ((mac_core_tx_crc_crc_prev[12] ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]);
	mac_core_tx_crc_crc_next[21] <= ((mac_core_tx_crc_crc_prev[13] ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]);
	mac_core_tx_crc_crc_next[22] <= ((mac_core_tx_crc_crc_prev[14] ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[23] <= ((((((mac_core_tx_crc_crc_prev[15] ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_data1[6]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[24] <= ((((((mac_core_tx_crc_crc_prev[16] ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]);
	mac_core_tx_crc_crc_next[25] <= ((((mac_core_tx_crc_crc_prev[17] ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]);
	mac_core_tx_crc_crc_next[26] <= ((((((((mac_core_tx_crc_crc_prev[18] ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]) ^ mac_core_tx_crc_crc_prev[24]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_data1[7]);
	mac_core_tx_crc_crc_next[27] <= ((((((((mac_core_tx_crc_crc_prev[19] ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]) ^ mac_core_tx_crc_crc_prev[25]) ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_data1[6]);
	mac_core_tx_crc_crc_next[28] <= ((((((mac_core_tx_crc_crc_prev[20] ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]) ^ mac_core_tx_crc_crc_prev[26]) ^ mac_core_tx_crc_data1[5]);
	mac_core_tx_crc_crc_next[29] <= ((((((mac_core_tx_crc_crc_prev[21] ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_crc_prev[30]) ^ mac_core_tx_crc_data1[1]) ^ mac_core_tx_crc_crc_prev[27]) ^ mac_core_tx_crc_data1[4]);
	mac_core_tx_crc_crc_next[30] <= ((((mac_core_tx_crc_crc_prev[22] ^ mac_core_tx_crc_crc_prev[31]) ^ mac_core_tx_crc_data1[0]) ^ mac_core_tx_crc_crc_prev[28]) ^ mac_core_tx_crc_data1[3]);
	mac_core_tx_crc_crc_next[31] <= ((mac_core_tx_crc_crc_prev[23] ^ mac_core_tx_crc_crc_prev[29]) ^ mac_core_tx_crc_data1[2]);
// synthesis translate_off
	dummy_d_2 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_3;
// synthesis translate_on
always @(*) begin
	mac_core_tx_crc_sink_ready <= 1'd0;
	mac_core_tx_crc_source_valid <= 1'd0;
	mac_core_tx_crc_source_first <= 1'd0;
	mac_core_tx_crc_source_last <= 1'd0;
	mac_core_tx_crc_source_payload_data <= 8'd0;
	mac_core_tx_crc_source_payload_be <= 1'd0;
	mac_core_tx_crc_source_payload_error <= 1'd0;
	mac_core_tx_crc_ce <= 1'd0;
	mac_core_tx_crc_reset <= 1'd0;
	mac_core_tx_crc_is_ongoing0 <= 1'd0;
	mac_core_tx_crc_is_ongoing1 <= 1'd0;
	liteethmac_txdatapath_bufferizeendpoints_next_state <= 2'd0;
	mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value0 <= 32'd0;
	mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value_ce0 <= 1'd0;
	mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value1 <= 1'd0;
	mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value_ce1 <= 1'd0;
	liteethmac_txdatapath_bufferizeendpoints_next_state <= liteethmac_txdatapath_bufferizeendpoints_state;
	case (liteethmac_txdatapath_bufferizeendpoints_state)
		1'd1: begin
			mac_core_tx_crc_ce <= (mac_core_tx_crc_sink_valid & mac_core_tx_crc_source_ready);
			mac_core_tx_crc_source_valid <= mac_core_tx_crc_sink_valid;
			mac_core_tx_crc_sink_ready <= mac_core_tx_crc_source_ready;
			mac_core_tx_crc_source_first <= mac_core_tx_crc_sink_first;
			mac_core_tx_crc_source_last <= mac_core_tx_crc_sink_last;
			mac_core_tx_crc_source_payload_data <= mac_core_tx_crc_sink_payload_data;
			mac_core_tx_crc_source_payload_be <= mac_core_tx_crc_sink_payload_be;
			mac_core_tx_crc_source_payload_error <= mac_core_tx_crc_sink_payload_error;
			mac_core_tx_crc_source_last <= 1'd0;
			mac_core_tx_crc_source_payload_be <= 1'd1;
			if (mac_core_tx_crc_sink_last) begin
				if (mac_core_tx_crc_sink_payload_be) begin
					mac_core_tx_crc_source_payload_data <= complexslicelowerer_slice_proxy[7:0];
				end
				if ((1'd0 & (mac_core_tx_crc_sink_payload_be <= 4'd15))) begin
					mac_core_tx_crc_source_last <= 1'd1;
					mac_core_tx_crc_source_payload_be <= ((mac_core_tx_crc_sink_payload_be <<< 3'd4) | 4'd15);
				end
			end
			if (((mac_core_tx_crc_sink_valid & mac_core_tx_crc_sink_last) & mac_core_tx_crc_source_ready)) begin
				if ((1'd0 & (mac_core_tx_crc_sink_payload_be <= 4'd15))) begin
					liteethmac_txdatapath_bufferizeendpoints_next_state <= 1'd0;
				end else begin
					mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value0 <= mac_core_tx_crc_value;
					mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value_ce0 <= 1'd1;
					if (1'd0) begin
						mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value1 <= (mac_core_tx_crc_sink_payload_be >>> 3'd4);
						mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value_ce1 <= 1'd1;
					end else begin
						mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value1 <= mac_core_tx_crc_sink_payload_be;
						mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value_ce1 <= 1'd1;
					end
					liteethmac_txdatapath_bufferizeendpoints_next_state <= 2'd2;
				end
			end
		end
		2'd2: begin
			mac_core_tx_crc_source_valid <= 1'd1;
			mac_core_tx_crc_source_payload_be <= 1'd1;
			case (mac_core_tx_crc_cnt)
				1'd0: begin
					mac_core_tx_crc_source_payload_data <= mac_core_tx_crc_crc_packet[31:24];
				end
				1'd1: begin
					mac_core_tx_crc_source_payload_data <= mac_core_tx_crc_crc_packet[23:16];
				end
				2'd2: begin
					mac_core_tx_crc_source_payload_data <= mac_core_tx_crc_crc_packet[15:8];
				end
				default: begin
					mac_core_tx_crc_source_payload_data <= mac_core_tx_crc_crc_packet[7:0];
				end
			endcase
			if (mac_core_tx_crc_cnt_done) begin
				mac_core_tx_crc_source_last <= 1'd1;
				if (mac_core_tx_crc_source_ready) begin
					liteethmac_txdatapath_bufferizeendpoints_next_state <= 1'd0;
				end
			end
			mac_core_tx_crc_is_ongoing1 <= 1'd1;
		end
		default: begin
			mac_core_tx_crc_reset <= 1'd1;
			mac_core_tx_crc_sink_ready <= 1'd1;
			if (mac_core_tx_crc_sink_valid) begin
				mac_core_tx_crc_sink_ready <= 1'd0;
				liteethmac_txdatapath_bufferizeendpoints_next_state <= 1'd1;
			end
			mac_core_tx_crc_is_ongoing0 <= 1'd1;
		end
	endcase
// synthesis translate_off
	dummy_d_3 <= dummy_s;
// synthesis translate_on
end
assign mac_core_tx_crc_pipe_valid_sink_ready = ((~mac_core_tx_crc_pipe_valid_source_valid) | mac_core_tx_crc_pipe_valid_source_ready);
assign mac_core_tx_crc_pipe_valid_sink_valid = mac_core_tx_crc_sink_sink_valid;
assign mac_core_tx_crc_sink_sink_ready = mac_core_tx_crc_pipe_valid_sink_ready;
assign mac_core_tx_crc_pipe_valid_sink_first = mac_core_tx_crc_sink_sink_first;
assign mac_core_tx_crc_pipe_valid_sink_last = mac_core_tx_crc_sink_sink_last;
assign mac_core_tx_crc_pipe_valid_sink_payload_data = mac_core_tx_crc_sink_sink_payload_data;
assign mac_core_tx_crc_pipe_valid_sink_payload_be = mac_core_tx_crc_sink_sink_payload_be;
assign mac_core_tx_crc_pipe_valid_sink_payload_error = mac_core_tx_crc_sink_sink_payload_error;
assign mac_core_tx_crc_source_source_valid = mac_core_tx_crc_pipe_valid_source_valid;
assign mac_core_tx_crc_pipe_valid_source_ready = mac_core_tx_crc_source_source_ready;
assign mac_core_tx_crc_source_source_first = mac_core_tx_crc_pipe_valid_source_first;
assign mac_core_tx_crc_source_source_last = mac_core_tx_crc_pipe_valid_source_last;
assign mac_core_tx_crc_source_source_payload_data = mac_core_tx_crc_pipe_valid_source_payload_data;
assign mac_core_tx_crc_source_source_payload_be = mac_core_tx_crc_pipe_valid_source_payload_be;
assign mac_core_tx_crc_source_source_payload_error = mac_core_tx_crc_pipe_valid_source_payload_error;

// synthesis translate_off
reg dummy_d_4;
// synthesis translate_on
always @(*) begin
	mac_core_tx_preamble_sink_ready <= 1'd0;
	mac_core_tx_preamble_source_valid <= 1'd0;
	mac_core_tx_preamble_source_first <= 1'd0;
	mac_core_tx_preamble_source_last <= 1'd0;
	mac_core_tx_preamble_source_payload_data <= 8'd0;
	mac_core_tx_preamble_source_payload_be <= 1'd0;
	mac_core_tx_preamble_source_payload_error <= 1'd0;
	liteethmac_txdatapath_liteethmacpreambleinserter_next_state <= 2'd0;
	mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value <= 3'd0;
	mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value_ce <= 1'd0;
	mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_sink_payload_data;
	mac_core_tx_preamble_source_payload_be <= mac_core_tx_preamble_sink_payload_be;
	liteethmac_txdatapath_liteethmacpreambleinserter_next_state <= liteethmac_txdatapath_liteethmacpreambleinserter_state;
	case (liteethmac_txdatapath_liteethmacpreambleinserter_state)
		1'd1: begin
			mac_core_tx_preamble_source_valid <= 1'd1;
			mac_core_tx_preamble_source_payload_be <= 1'd1;
			case (mac_core_tx_preamble_count)
				1'd0: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[7:0];
				end
				1'd1: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[15:8];
				end
				2'd2: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[23:16];
				end
				2'd3: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[31:24];
				end
				3'd4: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[39:32];
				end
				3'd5: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[47:40];
				end
				3'd6: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[55:48];
				end
				default: begin
					mac_core_tx_preamble_source_payload_data <= mac_core_tx_preamble_preamble[63:56];
				end
			endcase
			if (mac_core_tx_preamble_source_ready) begin
				if ((mac_core_tx_preamble_count == 3'd7)) begin
					liteethmac_txdatapath_liteethmacpreambleinserter_next_state <= 2'd2;
				end else begin
					mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value <= (mac_core_tx_preamble_count + 1'd1);
					mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value_ce <= 1'd1;
				end
			end
		end
		2'd2: begin
			mac_core_tx_preamble_source_valid <= mac_core_tx_preamble_sink_valid;
			mac_core_tx_preamble_sink_ready <= mac_core_tx_preamble_source_ready;
			mac_core_tx_preamble_source_first <= mac_core_tx_preamble_sink_first;
			mac_core_tx_preamble_source_last <= mac_core_tx_preamble_sink_last;
			mac_core_tx_preamble_source_payload_error <= mac_core_tx_preamble_sink_payload_error;
			if (((mac_core_tx_preamble_sink_valid & mac_core_tx_preamble_sink_last) & mac_core_tx_preamble_source_ready)) begin
				liteethmac_txdatapath_liteethmacpreambleinserter_next_state <= 1'd0;
			end
		end
		default: begin
			mac_core_tx_preamble_sink_ready <= 1'd1;
			mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value <= 1'd0;
			mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value_ce <= 1'd1;
			if (mac_core_tx_preamble_sink_valid) begin
				mac_core_tx_preamble_sink_ready <= 1'd0;
				liteethmac_txdatapath_liteethmacpreambleinserter_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_4 <= dummy_s;
// synthesis translate_on
end
assign mac_core_txdatapath_sink_sink_valid = mac_core_txdatapath_sink_valid;
assign mac_core_txdatapath_sink_ready = mac_core_txdatapath_sink_sink_ready;
assign mac_core_txdatapath_sink_sink_first = mac_core_txdatapath_sink_first;
assign mac_core_txdatapath_sink_sink_last = mac_core_txdatapath_sink_last;
assign mac_core_txdatapath_sink_sink_payload_data = mac_core_txdatapath_sink_payload_data;
assign mac_core_txdatapath_sink_sink_payload_be = mac_core_txdatapath_sink_payload_be;
assign mac_core_txdatapath_sink_sink_payload_error = mac_core_txdatapath_sink_payload_error;
assign mac_core_txdatapath_source_valid = mac_core_txdatapath_source_source_valid;
assign mac_core_txdatapath_source_source_ready = mac_core_txdatapath_source_ready;
assign mac_core_txdatapath_source_first = mac_core_txdatapath_source_source_first;
assign mac_core_txdatapath_source_last = mac_core_txdatapath_source_source_last;
assign mac_core_txdatapath_source_payload_data = mac_core_txdatapath_source_source_payload_data;
assign mac_core_txdatapath_source_payload_be = mac_core_txdatapath_source_source_payload_be;
assign mac_core_txdatapath_source_payload_error = mac_core_txdatapath_source_source_payload_error;
assign mac_core_txdatapath_pipe_valid_sink_ready = ((~mac_core_txdatapath_pipe_valid_source_valid) | mac_core_txdatapath_pipe_valid_source_ready);
assign mac_core_txdatapath_pipe_valid_sink_valid = mac_core_txdatapath_sink_sink_valid;
assign mac_core_txdatapath_sink_sink_ready = mac_core_txdatapath_pipe_valid_sink_ready;
assign mac_core_txdatapath_pipe_valid_sink_first = mac_core_txdatapath_sink_sink_first;
assign mac_core_txdatapath_pipe_valid_sink_last = mac_core_txdatapath_sink_sink_last;
assign mac_core_txdatapath_pipe_valid_sink_payload_data = mac_core_txdatapath_sink_sink_payload_data;
assign mac_core_txdatapath_pipe_valid_sink_payload_be = mac_core_txdatapath_sink_sink_payload_be;
assign mac_core_txdatapath_pipe_valid_sink_payload_error = mac_core_txdatapath_sink_sink_payload_error;
assign mac_core_txdatapath_source_source_valid = mac_core_txdatapath_pipe_valid_source_valid;
assign mac_core_txdatapath_pipe_valid_source_ready = mac_core_txdatapath_source_source_ready;
assign mac_core_txdatapath_source_source_first = mac_core_txdatapath_pipe_valid_source_first;
assign mac_core_txdatapath_source_source_last = mac_core_txdatapath_pipe_valid_source_last;
assign mac_core_txdatapath_source_source_payload_data = mac_core_txdatapath_pipe_valid_source_payload_data;
assign mac_core_txdatapath_source_source_payload_be = mac_core_txdatapath_pipe_valid_source_payload_be;
assign mac_core_txdatapath_source_source_payload_error = mac_core_txdatapath_pipe_valid_source_payload_error;

// synthesis translate_off
reg dummy_d_5;
// synthesis translate_on
always @(*) begin
	mac_core_tx_gap_sink_ready <= 1'd0;
	mac_core_tx_gap_source_valid <= 1'd0;
	mac_core_tx_gap_source_first <= 1'd0;
	mac_core_tx_gap_source_last <= 1'd0;
	mac_core_tx_gap_source_payload_data <= 8'd0;
	mac_core_tx_gap_source_payload_be <= 1'd0;
	mac_core_tx_gap_source_payload_error <= 1'd0;
	liteethmac_txdatapath_resetinserter_next_state <= 1'd0;
	mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value <= 4'd0;
	mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value_ce <= 1'd0;
	liteethmac_txdatapath_resetinserter_next_state <= liteethmac_txdatapath_resetinserter_state;
	case (liteethmac_txdatapath_resetinserter_state)
		1'd1: begin
			mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value <= (mac_core_tx_gap_counter - 1'd1);
			mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value_ce <= 1'd1;
			if ((mac_core_tx_gap_counter == 1'd1)) begin
				liteethmac_txdatapath_resetinserter_next_state <= 1'd0;
			end
		end
		default: begin
			mac_core_tx_gap_source_valid <= mac_core_tx_gap_sink_valid;
			mac_core_tx_gap_sink_ready <= mac_core_tx_gap_source_ready;
			mac_core_tx_gap_source_first <= mac_core_tx_gap_sink_first;
			mac_core_tx_gap_source_last <= mac_core_tx_gap_sink_last;
			mac_core_tx_gap_source_payload_data <= mac_core_tx_gap_sink_payload_data;
			mac_core_tx_gap_source_payload_be <= mac_core_tx_gap_sink_payload_be;
			mac_core_tx_gap_source_payload_error <= mac_core_tx_gap_sink_payload_error;
			if (((mac_core_tx_gap_sink_valid & mac_core_tx_gap_sink_last) & mac_core_tx_gap_sink_ready)) begin
				mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value <= 4'd12;
				mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value_ce <= 1'd1;
				liteethmac_txdatapath_resetinserter_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_5 <= dummy_s;
// synthesis translate_on
end
assign mac_core_tx_padding_sink_valid = mac_core_sink_valid;
assign mac_core_sink_ready = mac_core_tx_padding_sink_ready;
assign mac_core_tx_padding_sink_first = mac_core_sink_first;
assign mac_core_tx_padding_sink_last = mac_core_sink_last;
assign mac_core_tx_padding_sink_payload_data = mac_core_sink_payload_data;
assign mac_core_tx_padding_sink_payload_be = mac_core_sink_payload_be;
assign mac_core_tx_padding_sink_payload_error = mac_core_sink_payload_error;
assign mac_core_tx_crc_sink_sink_valid = mac_core_tx_padding_source_valid;
assign mac_core_tx_padding_source_ready = mac_core_tx_crc_sink_sink_ready;
assign mac_core_tx_crc_sink_sink_first = mac_core_tx_padding_source_first;
assign mac_core_tx_crc_sink_sink_last = mac_core_tx_padding_source_last;
assign mac_core_tx_crc_sink_sink_payload_data = mac_core_tx_padding_source_payload_data;
assign mac_core_tx_crc_sink_sink_payload_be = mac_core_tx_padding_source_payload_be;
assign mac_core_tx_crc_sink_sink_payload_error = mac_core_tx_padding_source_payload_error;
assign mac_core_tx_preamble_sink_valid = mac_core_tx_crc_source_valid;
assign mac_core_tx_crc_source_ready = mac_core_tx_preamble_sink_ready;
assign mac_core_tx_preamble_sink_first = mac_core_tx_crc_source_first;
assign mac_core_tx_preamble_sink_last = mac_core_tx_crc_source_last;
assign mac_core_tx_preamble_sink_payload_data = mac_core_tx_crc_source_payload_data;
assign mac_core_tx_preamble_sink_payload_be = mac_core_tx_crc_source_payload_be;
assign mac_core_tx_preamble_sink_payload_error = mac_core_tx_crc_source_payload_error;
assign mac_core_txdatapath_sink_valid = mac_core_tx_preamble_source_valid;
assign mac_core_tx_preamble_source_ready = mac_core_txdatapath_sink_ready;
assign mac_core_txdatapath_sink_first = mac_core_tx_preamble_source_first;
assign mac_core_txdatapath_sink_last = mac_core_tx_preamble_source_last;
assign mac_core_txdatapath_sink_payload_data = mac_core_tx_preamble_source_payload_data;
assign mac_core_txdatapath_sink_payload_be = mac_core_tx_preamble_source_payload_be;
assign mac_core_txdatapath_sink_payload_error = mac_core_tx_preamble_source_payload_error;
assign mac_core_tx_gap_sink_valid = mac_core_txdatapath_source_valid;
assign mac_core_txdatapath_source_ready = mac_core_tx_gap_sink_ready;
assign mac_core_tx_gap_sink_first = mac_core_txdatapath_source_first;
assign mac_core_tx_gap_sink_last = mac_core_txdatapath_source_last;
assign mac_core_tx_gap_sink_payload_data = mac_core_txdatapath_source_payload_data;
assign mac_core_tx_gap_sink_payload_be = mac_core_txdatapath_source_payload_be;
assign mac_core_tx_gap_sink_payload_error = mac_core_txdatapath_source_payload_error;
assign phy_sink_valid = mac_core_tx_gap_source_valid;
assign mac_core_tx_gap_source_ready = phy_sink_ready;
assign phy_sink_first = mac_core_tx_gap_source_first;
assign phy_sink_last = mac_core_tx_gap_source_last;
assign phy_sink_payload_data = mac_core_tx_gap_source_payload_data;
assign phy_sink_payload_be = mac_core_tx_gap_source_payload_be;
assign phy_sink_payload_error = mac_core_tx_gap_source_payload_error;
assign mac_core_pulsesynchronizer0_i = mac_core_rx_preamble_error;
assign mac_core_pulsesynchronizer1_i = mac_core_liteethmaccrc32checker_error;
assign mac_core_rxdatapath_sink_sink_valid = mac_core_rxdatapath_sink_valid;
assign mac_core_rxdatapath_sink_ready = mac_core_rxdatapath_sink_sink_ready;
assign mac_core_rxdatapath_sink_sink_first = mac_core_rxdatapath_sink_first;
assign mac_core_rxdatapath_sink_sink_last = mac_core_rxdatapath_sink_last;
assign mac_core_rxdatapath_sink_sink_payload_data = mac_core_rxdatapath_sink_payload_data;
assign mac_core_rxdatapath_sink_sink_payload_be = mac_core_rxdatapath_sink_payload_be;
assign mac_core_rxdatapath_sink_sink_payload_error = mac_core_rxdatapath_sink_payload_error;
assign mac_core_rxdatapath_source_valid = mac_core_rxdatapath_source_source_valid;
assign mac_core_rxdatapath_source_source_ready = mac_core_rxdatapath_source_ready;
assign mac_core_rxdatapath_source_first = mac_core_rxdatapath_source_source_first;
assign mac_core_rxdatapath_source_last = mac_core_rxdatapath_source_source_last;
assign mac_core_rxdatapath_source_payload_data = mac_core_rxdatapath_source_source_payload_data;
assign mac_core_rxdatapath_source_payload_be = mac_core_rxdatapath_source_source_payload_be;
assign mac_core_rxdatapath_source_payload_error = mac_core_rxdatapath_source_source_payload_error;
assign mac_core_rxdatapath_pipe_valid_sink_ready = ((~mac_core_rxdatapath_pipe_valid_source_valid) | mac_core_rxdatapath_pipe_valid_source_ready);
assign mac_core_rxdatapath_pipe_valid_sink_valid = mac_core_rxdatapath_sink_sink_valid;
assign mac_core_rxdatapath_sink_sink_ready = mac_core_rxdatapath_pipe_valid_sink_ready;
assign mac_core_rxdatapath_pipe_valid_sink_first = mac_core_rxdatapath_sink_sink_first;
assign mac_core_rxdatapath_pipe_valid_sink_last = mac_core_rxdatapath_sink_sink_last;
assign mac_core_rxdatapath_pipe_valid_sink_payload_data = mac_core_rxdatapath_sink_sink_payload_data;
assign mac_core_rxdatapath_pipe_valid_sink_payload_be = mac_core_rxdatapath_sink_sink_payload_be;
assign mac_core_rxdatapath_pipe_valid_sink_payload_error = mac_core_rxdatapath_sink_sink_payload_error;
assign mac_core_rxdatapath_source_source_valid = mac_core_rxdatapath_pipe_valid_source_valid;
assign mac_core_rxdatapath_pipe_valid_source_ready = mac_core_rxdatapath_source_source_ready;
assign mac_core_rxdatapath_source_source_first = mac_core_rxdatapath_pipe_valid_source_first;
assign mac_core_rxdatapath_source_source_last = mac_core_rxdatapath_pipe_valid_source_last;
assign mac_core_rxdatapath_source_source_payload_data = mac_core_rxdatapath_pipe_valid_source_payload_data;
assign mac_core_rxdatapath_source_source_payload_be = mac_core_rxdatapath_pipe_valid_source_payload_be;
assign mac_core_rxdatapath_source_source_payload_error = mac_core_rxdatapath_pipe_valid_source_payload_error;
assign mac_core_rx_preamble_source_payload_data = mac_core_rx_preamble_sink_payload_data;
assign mac_core_rx_preamble_source_payload_be = mac_core_rx_preamble_sink_payload_be;

// synthesis translate_off
reg dummy_d_6;
// synthesis translate_on
always @(*) begin
	mac_core_rx_preamble_sink_ready <= 1'd0;
	mac_core_rx_preamble_source_valid <= 1'd0;
	mac_core_rx_preamble_source_first <= 1'd0;
	mac_core_rx_preamble_source_last <= 1'd0;
	mac_core_rx_preamble_source_payload_error <= 1'd0;
	mac_core_rx_preamble_error <= 1'd0;
	liteethmac_rxdatapath_liteethmacpreamblechecker_next_state <= 1'd0;
	liteethmac_rxdatapath_liteethmacpreamblechecker_next_state <= liteethmac_rxdatapath_liteethmacpreamblechecker_state;
	case (liteethmac_rxdatapath_liteethmacpreamblechecker_state)
		1'd1: begin
			mac_core_rx_preamble_source_valid <= mac_core_rx_preamble_sink_valid;
			mac_core_rx_preamble_sink_ready <= mac_core_rx_preamble_source_ready;
			mac_core_rx_preamble_source_first <= mac_core_rx_preamble_sink_first;
			mac_core_rx_preamble_source_last <= mac_core_rx_preamble_sink_last;
			mac_core_rx_preamble_source_payload_error <= mac_core_rx_preamble_sink_payload_error;
			if (((mac_core_rx_preamble_source_valid & mac_core_rx_preamble_source_last) & mac_core_rx_preamble_source_ready)) begin
				liteethmac_rxdatapath_liteethmacpreamblechecker_next_state <= 1'd0;
			end
		end
		default: begin
			mac_core_rx_preamble_sink_ready <= 1'd1;
			if (((mac_core_rx_preamble_sink_valid & (~mac_core_rx_preamble_sink_last)) & (mac_core_rx_preamble_sink_payload_data == mac_core_rx_preamble_preamble[63:56]))) begin
				liteethmac_rxdatapath_liteethmacpreamblechecker_next_state <= 1'd1;
			end
			if ((mac_core_rx_preamble_sink_valid & mac_core_rx_preamble_sink_last)) begin
				mac_core_rx_preamble_error <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_6 <= dummy_s;
// synthesis translate_on
end
assign mac_core_pulsesynchronizer0_o = (mac_core_pulsesynchronizer0_toggle_o ^ mac_core_pulsesynchronizer0_toggle_o_r);
assign mac_core_liteethmaccrc32checker_fifo_full = (mac_core_liteethmaccrc32checker_syncfifo_level == 3'd4);
assign mac_core_liteethmaccrc32checker_fifo_out = (mac_core_liteethmaccrc32checker_source_source_valid & mac_core_liteethmaccrc32checker_source_source_ready);
assign mac_core_liteethmaccrc32checker_syncfifo_sink_first = mac_core_liteethmaccrc32checker_sink_sink_first;
assign mac_core_liteethmaccrc32checker_syncfifo_sink_last = mac_core_liteethmaccrc32checker_sink_sink_last;
assign mac_core_liteethmaccrc32checker_syncfifo_sink_payload_data = mac_core_liteethmaccrc32checker_sink_sink_payload_data;
assign mac_core_liteethmaccrc32checker_syncfifo_sink_payload_be = mac_core_liteethmaccrc32checker_sink_sink_payload_be;
assign mac_core_liteethmaccrc32checker_syncfifo_sink_payload_error = mac_core_liteethmaccrc32checker_sink_sink_payload_error;

// synthesis translate_off
reg dummy_d_7;
// synthesis translate_on
always @(*) begin
	mac_core_liteethmaccrc32checker_syncfifo_sink_valid <= 1'd0;
	mac_core_liteethmaccrc32checker_syncfifo_sink_valid <= mac_core_liteethmaccrc32checker_sink_sink_valid;
	mac_core_liteethmaccrc32checker_syncfifo_sink_valid <= mac_core_liteethmaccrc32checker_fifo_in;
// synthesis translate_off
	dummy_d_7 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_8;
// synthesis translate_on
always @(*) begin
	mac_core_liteethmaccrc32checker_sink_sink_ready <= 1'd0;
	mac_core_liteethmaccrc32checker_sink_sink_ready <= mac_core_liteethmaccrc32checker_syncfifo_sink_ready;
	mac_core_liteethmaccrc32checker_sink_sink_ready <= mac_core_liteethmaccrc32checker_fifo_in;
// synthesis translate_off
	dummy_d_8 <= dummy_s;
// synthesis translate_on
end
assign mac_core_liteethmaccrc32checker_fifo_in = (((mac_core_liteethmaccrc32checker_sink_sink_valid & ((~mac_core_liteethmaccrc32checker_fifo_full) | mac_core_liteethmaccrc32checker_fifo_out)) & (~mac_core_liteethmaccrc32checker_is_ongoing0)) & (~mac_core_liteethmaccrc32checker_is_ongoing1));
assign mac_core_liteethmaccrc32checker_crc_data0 = mac_core_liteethmaccrc32checker_sink_sink_payload_data;
assign mac_core_liteethmaccrc32checker_crc_be = mac_core_liteethmaccrc32checker_sink_sink_payload_be;
assign mac_core_liteethmaccrc32checker_last_word_error = ((mac_core_liteethmaccrc32checker_crc_error | mac_core_liteethmaccrc32checker_packet_error) | ((mac_core_liteethmaccrc32checker_sink_sink_payload_error & mac_core_liteethmaccrc32checker_sink_sink_payload_be) != 1'd0));
assign mac_core_liteethmaccrc32checker_source_source_first = mac_core_liteethmaccrc32checker_syncfifo_source_first;
assign mac_core_liteethmaccrc32checker_source_source_payload_data = mac_core_liteethmaccrc32checker_syncfifo_source_payload_data;
assign mac_core_liteethmaccrc32checker_sink_sink_valid = mac_core_bufferizeendpoints_source_source_valid;
assign mac_core_bufferizeendpoints_source_source_ready = mac_core_liteethmaccrc32checker_sink_sink_ready;
assign mac_core_liteethmaccrc32checker_sink_sink_first = mac_core_bufferizeendpoints_source_source_first;
assign mac_core_liteethmaccrc32checker_sink_sink_last = mac_core_bufferizeendpoints_source_source_last;
assign mac_core_liteethmaccrc32checker_sink_sink_payload_data = mac_core_bufferizeendpoints_source_source_payload_data;
assign mac_core_liteethmaccrc32checker_sink_sink_payload_be = mac_core_bufferizeendpoints_source_source_payload_be;
assign mac_core_liteethmaccrc32checker_sink_sink_payload_error = mac_core_bufferizeendpoints_source_source_payload_error;
assign mac_core_liteethmaccrc32checker_crc_crc_prev = mac_core_liteethmaccrc32checker_crc_reg;

// synthesis translate_off
reg dummy_d_9;
// synthesis translate_on
always @(*) begin
	mac_core_liteethmaccrc32checker_crc_error <= 1'd0;
	mac_core_liteethmaccrc32checker_crc_data1 <= 8'd0;
	mac_core_liteethmaccrc32checker_crc_data1 <= mac_core_liteethmaccrc32checker_crc_data0;
	if (mac_core_liteethmaccrc32checker_crc_be) begin
		mac_core_liteethmaccrc32checker_crc_data1 <= (mac_core_liteethmaccrc32checker_crc_data0 & 8'd255);
		mac_core_liteethmaccrc32checker_crc_error <= (mac_core_liteethmaccrc32checker_crc_crc_next != 32'd3338984827);
	end
// synthesis translate_off
	dummy_d_9 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_10;
// synthesis translate_on
always @(*) begin
	mac_core_liteethmaccrc32checker_crc_crc_next <= 32'd0;
	mac_core_liteethmaccrc32checker_crc_crc_next[0] <= (((mac_core_liteethmaccrc32checker_crc_crc_prev[24] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[1] <= (((((((mac_core_liteethmaccrc32checker_crc_crc_prev[25] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[2] <= (((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[26] ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[3] <= (((((((mac_core_liteethmaccrc32checker_crc_crc_prev[27] ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]);
	mac_core_liteethmaccrc32checker_crc_crc_next[4] <= (((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[28] ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[5] <= (((((((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[29] ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[6] <= (((((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[30] ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]);
	mac_core_liteethmaccrc32checker_crc_crc_next[7] <= (((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[31] ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[8] <= ((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[0] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[9] <= ((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[1] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]);
	mac_core_liteethmaccrc32checker_crc_crc_next[10] <= ((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[2] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[11] <= ((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[3] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[12] <= ((((((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[4] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[13] <= ((((((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[5] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]);
	mac_core_liteethmaccrc32checker_crc_crc_next[14] <= ((((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[6] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]);
	mac_core_liteethmaccrc32checker_crc_crc_next[15] <= ((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[7] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]);
	mac_core_liteethmaccrc32checker_crc_crc_next[16] <= ((((((mac_core_liteethmaccrc32checker_crc_crc_prev[8] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[17] <= ((((((mac_core_liteethmaccrc32checker_crc_crc_prev[9] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]);
	mac_core_liteethmaccrc32checker_crc_crc_next[18] <= ((((((mac_core_liteethmaccrc32checker_crc_crc_prev[10] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]);
	mac_core_liteethmaccrc32checker_crc_crc_next[19] <= ((((mac_core_liteethmaccrc32checker_crc_crc_prev[11] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]);
	mac_core_liteethmaccrc32checker_crc_crc_next[20] <= ((mac_core_liteethmaccrc32checker_crc_crc_prev[12] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]);
	mac_core_liteethmaccrc32checker_crc_crc_next[21] <= ((mac_core_liteethmaccrc32checker_crc_crc_prev[13] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]);
	mac_core_liteethmaccrc32checker_crc_crc_next[22] <= ((mac_core_liteethmaccrc32checker_crc_crc_prev[14] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[23] <= ((((((mac_core_liteethmaccrc32checker_crc_crc_prev[15] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[24] <= ((((((mac_core_liteethmaccrc32checker_crc_crc_prev[16] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]);
	mac_core_liteethmaccrc32checker_crc_crc_next[25] <= ((((mac_core_liteethmaccrc32checker_crc_crc_prev[17] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]);
	mac_core_liteethmaccrc32checker_crc_crc_next[26] <= ((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[18] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[24]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_data1[7]);
	mac_core_liteethmaccrc32checker_crc_crc_next[27] <= ((((((((mac_core_liteethmaccrc32checker_crc_crc_prev[19] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[25]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_data1[6]);
	mac_core_liteethmaccrc32checker_crc_crc_next[28] <= ((((((mac_core_liteethmaccrc32checker_crc_crc_prev[20] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[26]) ^ mac_core_liteethmaccrc32checker_crc_data1[5]);
	mac_core_liteethmaccrc32checker_crc_crc_next[29] <= ((((((mac_core_liteethmaccrc32checker_crc_crc_prev[21] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[30]) ^ mac_core_liteethmaccrc32checker_crc_data1[1]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[27]) ^ mac_core_liteethmaccrc32checker_crc_data1[4]);
	mac_core_liteethmaccrc32checker_crc_crc_next[30] <= ((((mac_core_liteethmaccrc32checker_crc_crc_prev[22] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[31]) ^ mac_core_liteethmaccrc32checker_crc_data1[0]) ^ mac_core_liteethmaccrc32checker_crc_crc_prev[28]) ^ mac_core_liteethmaccrc32checker_crc_data1[3]);
	mac_core_liteethmaccrc32checker_crc_crc_next[31] <= ((mac_core_liteethmaccrc32checker_crc_crc_prev[23] ^ mac_core_liteethmaccrc32checker_crc_crc_prev[29]) ^ mac_core_liteethmaccrc32checker_crc_data1[2]);
// synthesis translate_off
	dummy_d_10 <= dummy_s;
// synthesis translate_on
end
assign mac_core_liteethmaccrc32checker_syncfifo_syncfifo_din = {mac_core_liteethmaccrc32checker_syncfifo_fifo_in_last, mac_core_liteethmaccrc32checker_syncfifo_fifo_in_first, mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_error, mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_be, mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_data};
assign {mac_core_liteethmaccrc32checker_syncfifo_fifo_out_last, mac_core_liteethmaccrc32checker_syncfifo_fifo_out_first, mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_error, mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_be, mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_data} = mac_core_liteethmaccrc32checker_syncfifo_syncfifo_dout;
assign mac_core_liteethmaccrc32checker_syncfifo_sink_ready = mac_core_liteethmaccrc32checker_syncfifo_syncfifo_writable;
assign mac_core_liteethmaccrc32checker_syncfifo_syncfifo_we = mac_core_liteethmaccrc32checker_syncfifo_sink_valid;
assign mac_core_liteethmaccrc32checker_syncfifo_fifo_in_first = mac_core_liteethmaccrc32checker_syncfifo_sink_first;
assign mac_core_liteethmaccrc32checker_syncfifo_fifo_in_last = mac_core_liteethmaccrc32checker_syncfifo_sink_last;
assign mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_data = mac_core_liteethmaccrc32checker_syncfifo_sink_payload_data;
assign mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_be = mac_core_liteethmaccrc32checker_syncfifo_sink_payload_be;
assign mac_core_liteethmaccrc32checker_syncfifo_fifo_in_payload_error = mac_core_liteethmaccrc32checker_syncfifo_sink_payload_error;
assign mac_core_liteethmaccrc32checker_syncfifo_source_valid = mac_core_liteethmaccrc32checker_syncfifo_syncfifo_readable;
assign mac_core_liteethmaccrc32checker_syncfifo_source_first = mac_core_liteethmaccrc32checker_syncfifo_fifo_out_first;
assign mac_core_liteethmaccrc32checker_syncfifo_source_last = mac_core_liteethmaccrc32checker_syncfifo_fifo_out_last;
assign mac_core_liteethmaccrc32checker_syncfifo_source_payload_data = mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_data;
assign mac_core_liteethmaccrc32checker_syncfifo_source_payload_be = mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_be;
assign mac_core_liteethmaccrc32checker_syncfifo_source_payload_error = mac_core_liteethmaccrc32checker_syncfifo_fifo_out_payload_error;
assign mac_core_liteethmaccrc32checker_syncfifo_syncfifo_re = mac_core_liteethmaccrc32checker_syncfifo_source_ready;

// synthesis translate_off
reg dummy_d_11;
// synthesis translate_on
always @(*) begin
	mac_core_liteethmaccrc32checker_syncfifo_wrport_adr <= 3'd0;
	if (mac_core_liteethmaccrc32checker_syncfifo_replace) begin
		mac_core_liteethmaccrc32checker_syncfifo_wrport_adr <= (mac_core_liteethmaccrc32checker_syncfifo_produce - 1'd1);
	end else begin
		mac_core_liteethmaccrc32checker_syncfifo_wrport_adr <= mac_core_liteethmaccrc32checker_syncfifo_produce;
	end
// synthesis translate_off
	dummy_d_11 <= dummy_s;
// synthesis translate_on
end
assign mac_core_liteethmaccrc32checker_syncfifo_wrport_dat_w = mac_core_liteethmaccrc32checker_syncfifo_syncfifo_din;
assign mac_core_liteethmaccrc32checker_syncfifo_wrport_we = (mac_core_liteethmaccrc32checker_syncfifo_syncfifo_we & (mac_core_liteethmaccrc32checker_syncfifo_syncfifo_writable | mac_core_liteethmaccrc32checker_syncfifo_replace));
assign mac_core_liteethmaccrc32checker_syncfifo_do_read = (mac_core_liteethmaccrc32checker_syncfifo_syncfifo_readable & mac_core_liteethmaccrc32checker_syncfifo_syncfifo_re);
assign mac_core_liteethmaccrc32checker_syncfifo_rdport_adr = mac_core_liteethmaccrc32checker_syncfifo_consume;
assign mac_core_liteethmaccrc32checker_syncfifo_syncfifo_dout = mac_core_liteethmaccrc32checker_syncfifo_rdport_dat_r;
assign mac_core_liteethmaccrc32checker_syncfifo_syncfifo_writable = (mac_core_liteethmaccrc32checker_syncfifo_level != 3'd5);
assign mac_core_liteethmaccrc32checker_syncfifo_syncfifo_readable = (mac_core_liteethmaccrc32checker_syncfifo_level != 1'd0);

// synthesis translate_off
reg dummy_d_12;
// synthesis translate_on
always @(*) begin
	mac_core_liteethmaccrc32checker_source_source_valid <= 1'd0;
	mac_core_liteethmaccrc32checker_source_source_last <= 1'd0;
	mac_core_liteethmaccrc32checker_source_source_payload_be <= 1'd0;
	mac_core_liteethmaccrc32checker_source_source_payload_error <= 1'd0;
	mac_core_liteethmaccrc32checker_error <= 1'd0;
	mac_core_liteethmaccrc32checker_crc_ce <= 1'd0;
	mac_core_liteethmaccrc32checker_crc_reset <= 1'd0;
	mac_core_liteethmaccrc32checker_syncfifo_source_ready <= 1'd0;
	mac_core_liteethmaccrc32checker_fifo_reset <= 1'd0;
	mac_core_liteethmaccrc32checker_is_ongoing0 <= 1'd0;
	mac_core_liteethmaccrc32checker_is_ongoing1 <= 1'd0;
	mac_core_liteethmaccrc32checker_is_ongoing2 <= 1'd0;
	liteethmac_rxdatapath_bufferizeendpoints_next_state <= 2'd0;
	mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value0 <= 1'd0;
	mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value_ce0 <= 1'd0;
	mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value1 <= 1'd0;
	mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value_ce1 <= 1'd0;
	mac_core_liteethmaccrc32checker_source_source_payload_error <= mac_core_liteethmaccrc32checker_syncfifo_source_payload_error;
	liteethmac_rxdatapath_bufferizeendpoints_next_state <= liteethmac_rxdatapath_bufferizeendpoints_state;
	case (liteethmac_rxdatapath_bufferizeendpoints_state)
		1'd1: begin
			mac_core_liteethmaccrc32checker_is_ongoing1 <= 1'd1;
			mac_core_liteethmaccrc32checker_source_source_valid <= mac_core_liteethmaccrc32checker_syncfifo_source_valid;
			mac_core_liteethmaccrc32checker_syncfifo_source_ready <= mac_core_liteethmaccrc32checker_source_source_ready;
			mac_core_liteethmaccrc32checker_source_source_last <= mac_core_liteethmaccrc32checker_syncfifo_source_last;
			mac_core_liteethmaccrc32checker_source_source_payload_error <= (mac_core_liteethmaccrc32checker_syncfifo_source_payload_error | {1{mac_core_liteethmaccrc32checker_last_error}});
			mac_core_liteethmaccrc32checker_source_source_payload_be <= mac_core_liteethmaccrc32checker_be;
			if ((mac_core_liteethmaccrc32checker_source_source_valid & mac_core_liteethmaccrc32checker_source_source_ready)) begin
				liteethmac_rxdatapath_bufferizeendpoints_next_state <= 1'd0;
			end
		end
		2'd2: begin
			if ((mac_core_liteethmaccrc32checker_sink_sink_valid & mac_core_liteethmaccrc32checker_sink_sink_ready)) begin
				mac_core_liteethmaccrc32checker_crc_ce <= 1'd1;
				liteethmac_rxdatapath_bufferizeendpoints_next_state <= 2'd3;
			end
		end
		2'd3: begin
			mac_core_liteethmaccrc32checker_syncfifo_source_ready <= mac_core_liteethmaccrc32checker_fifo_out;
			mac_core_liteethmaccrc32checker_source_source_valid <= (mac_core_liteethmaccrc32checker_sink_sink_valid & mac_core_liteethmaccrc32checker_fifo_full);
			mac_core_liteethmaccrc32checker_source_source_payload_be <= 1'd1;
			if (1'd1) begin
				mac_core_liteethmaccrc32checker_source_source_last <= mac_core_liteethmaccrc32checker_sink_sink_last;
				if (mac_core_liteethmaccrc32checker_sink_sink_last) begin
					mac_core_liteethmaccrc32checker_source_source_payload_be <= mac_core_liteethmaccrc32checker_sink_sink_payload_be;
				end
			end else begin
				if ((mac_core_liteethmaccrc32checker_sink_sink_payload_be <= 4'd15)) begin
					mac_core_liteethmaccrc32checker_source_source_last <= mac_core_liteethmaccrc32checker_sink_sink_last;
					mac_core_liteethmaccrc32checker_source_source_payload_be <= ((mac_core_liteethmaccrc32checker_sink_sink_payload_be <<< 3'd4) | 4'd15);
				end else begin
					mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value0 <= (mac_core_liteethmaccrc32checker_sink_sink_payload_be >>> 3'd4);
					mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value_ce0 <= 1'd1;
					mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value1 <= mac_core_liteethmaccrc32checker_last_word_error;
					mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value_ce1 <= 1'd1;
				end
			end
			mac_core_liteethmaccrc32checker_source_source_payload_error <= (mac_core_liteethmaccrc32checker_sink_sink_last ? {1{mac_core_liteethmaccrc32checker_last_word_error}} : mac_core_liteethmaccrc32checker_syncfifo_source_payload_error);
			mac_core_liteethmaccrc32checker_error <= ((mac_core_liteethmaccrc32checker_sink_sink_valid & mac_core_liteethmaccrc32checker_sink_sink_last) & mac_core_liteethmaccrc32checker_crc_error);
			if ((mac_core_liteethmaccrc32checker_sink_sink_valid & mac_core_liteethmaccrc32checker_sink_sink_ready)) begin
				mac_core_liteethmaccrc32checker_crc_ce <= 1'd1;
				if ((mac_core_liteethmaccrc32checker_sink_sink_last & (mac_core_liteethmaccrc32checker_sink_sink_payload_be > 4'd15))) begin
					liteethmac_rxdatapath_bufferizeendpoints_next_state <= 1'd1;
				end else begin
					if (mac_core_liteethmaccrc32checker_sink_sink_last) begin
						liteethmac_rxdatapath_bufferizeendpoints_next_state <= 1'd0;
					end
				end
			end
		end
		default: begin
			mac_core_liteethmaccrc32checker_is_ongoing0 <= 1'd1;
			mac_core_liteethmaccrc32checker_crc_reset <= 1'd1;
			mac_core_liteethmaccrc32checker_fifo_reset <= 1'd1;
			liteethmac_rxdatapath_bufferizeendpoints_next_state <= 2'd2;
			mac_core_liteethmaccrc32checker_is_ongoing2 <= 1'd1;
		end
	endcase
// synthesis translate_off
	dummy_d_12 <= dummy_s;
// synthesis translate_on
end
assign mac_core_bufferizeendpoints_pipe_valid_sink_ready = ((~mac_core_bufferizeendpoints_pipe_valid_source_valid) | mac_core_bufferizeendpoints_pipe_valid_source_ready);
assign mac_core_bufferizeendpoints_pipe_valid_sink_valid = mac_core_bufferizeendpoints_sink_sink_valid;
assign mac_core_bufferizeendpoints_sink_sink_ready = mac_core_bufferizeendpoints_pipe_valid_sink_ready;
assign mac_core_bufferizeendpoints_pipe_valid_sink_first = mac_core_bufferizeendpoints_sink_sink_first;
assign mac_core_bufferizeendpoints_pipe_valid_sink_last = mac_core_bufferizeendpoints_sink_sink_last;
assign mac_core_bufferizeendpoints_pipe_valid_sink_payload_data = mac_core_bufferizeendpoints_sink_sink_payload_data;
assign mac_core_bufferizeendpoints_pipe_valid_sink_payload_be = mac_core_bufferizeendpoints_sink_sink_payload_be;
assign mac_core_bufferizeendpoints_pipe_valid_sink_payload_error = mac_core_bufferizeendpoints_sink_sink_payload_error;
assign mac_core_bufferizeendpoints_source_source_valid = mac_core_bufferizeendpoints_pipe_valid_source_valid;
assign mac_core_bufferizeendpoints_pipe_valid_source_ready = mac_core_bufferizeendpoints_source_source_ready;
assign mac_core_bufferizeendpoints_source_source_first = mac_core_bufferizeendpoints_pipe_valid_source_first;
assign mac_core_bufferizeendpoints_source_source_last = mac_core_bufferizeendpoints_pipe_valid_source_last;
assign mac_core_bufferizeendpoints_source_source_payload_data = mac_core_bufferizeendpoints_pipe_valid_source_payload_data;
assign mac_core_bufferizeendpoints_source_source_payload_be = mac_core_bufferizeendpoints_pipe_valid_source_payload_be;
assign mac_core_bufferizeendpoints_source_source_payload_error = mac_core_bufferizeendpoints_pipe_valid_source_payload_error;
assign mac_core_pulsesynchronizer1_o = (mac_core_pulsesynchronizer1_toggle_o ^ mac_core_pulsesynchronizer1_toggle_o_r);
assign mac_core_rx_padding_length_inc = (1'd0 + mac_core_rx_padding_sink_payload_be);
assign mac_core_rx_padding_source_valid = mac_core_rx_padding_sink_valid;
assign mac_core_rx_padding_sink_ready = mac_core_rx_padding_source_ready;
assign mac_core_rx_padding_source_first = mac_core_rx_padding_sink_first;
assign mac_core_rx_padding_source_last = mac_core_rx_padding_sink_last;
assign mac_core_rx_padding_source_payload_data = mac_core_rx_padding_sink_payload_data;
assign mac_core_rx_padding_source_payload_be = mac_core_rx_padding_sink_payload_be;

// synthesis translate_off
reg dummy_d_13;
// synthesis translate_on
always @(*) begin
	mac_core_rx_padding_source_payload_error <= 1'd0;
	if (((mac_core_rx_padding_sink_valid & mac_core_rx_padding_sink_last) & ((mac_core_rx_padding_length + mac_core_rx_padding_length_inc) < 6'd60))) begin
		mac_core_rx_padding_source_payload_error <= {1{1'd1}};
	end else begin
		mac_core_rx_padding_source_payload_error <= mac_core_rx_padding_sink_payload_error;
	end
// synthesis translate_off
	dummy_d_13 <= dummy_s;
// synthesis translate_on
end
assign mac_core_rxdatapath_sink_valid = phy_source_valid;
assign phy_source_ready = mac_core_rxdatapath_sink_ready;
assign mac_core_rxdatapath_sink_first = phy_source_first;
assign mac_core_rxdatapath_sink_last = phy_source_last;
assign mac_core_rxdatapath_sink_payload_data = phy_source_payload_data;
assign mac_core_rxdatapath_sink_payload_be = phy_source_payload_be;
assign mac_core_rxdatapath_sink_payload_error = phy_source_payload_error;
assign mac_core_rx_preamble_sink_valid = mac_core_rxdatapath_source_valid;
assign mac_core_rxdatapath_source_ready = mac_core_rx_preamble_sink_ready;
assign mac_core_rx_preamble_sink_first = mac_core_rxdatapath_source_first;
assign mac_core_rx_preamble_sink_last = mac_core_rxdatapath_source_last;
assign mac_core_rx_preamble_sink_payload_data = mac_core_rxdatapath_source_payload_data;
assign mac_core_rx_preamble_sink_payload_be = mac_core_rxdatapath_source_payload_be;
assign mac_core_rx_preamble_sink_payload_error = mac_core_rxdatapath_source_payload_error;
assign mac_core_bufferizeendpoints_sink_sink_valid = mac_core_rx_preamble_source_valid;
assign mac_core_rx_preamble_source_ready = mac_core_bufferizeendpoints_sink_sink_ready;
assign mac_core_bufferizeendpoints_sink_sink_first = mac_core_rx_preamble_source_first;
assign mac_core_bufferizeendpoints_sink_sink_last = mac_core_rx_preamble_source_last;
assign mac_core_bufferizeendpoints_sink_sink_payload_data = mac_core_rx_preamble_source_payload_data;
assign mac_core_bufferizeendpoints_sink_sink_payload_be = mac_core_rx_preamble_source_payload_be;
assign mac_core_bufferizeendpoints_sink_sink_payload_error = mac_core_rx_preamble_source_payload_error;
assign mac_core_rx_padding_sink_valid = mac_core_liteethmaccrc32checker_source_source_valid;
assign mac_core_liteethmaccrc32checker_source_source_ready = mac_core_rx_padding_sink_ready;
assign mac_core_rx_padding_sink_first = mac_core_liteethmaccrc32checker_source_source_first;
assign mac_core_rx_padding_sink_last = mac_core_liteethmaccrc32checker_source_source_last;
assign mac_core_rx_padding_sink_payload_data = mac_core_liteethmaccrc32checker_source_source_payload_data;
assign mac_core_rx_padding_sink_payload_be = mac_core_liteethmaccrc32checker_source_source_payload_be;
assign mac_core_rx_padding_sink_payload_error = mac_core_liteethmaccrc32checker_source_source_payload_error;
assign mac_core_source_valid = mac_core_rx_padding_source_valid;
assign mac_core_rx_padding_source_ready = mac_core_source_ready;
assign mac_core_source_first = mac_core_rx_padding_source_first;
assign mac_core_source_last = mac_core_rx_padding_source_last;
assign mac_core_source_payload_data = mac_core_rx_padding_source_payload_data;
assign mac_core_source_payload_be = mac_core_rx_padding_source_payload_be;
assign mac_core_source_payload_error = mac_core_rx_padding_source_payload_error;

// synthesis translate_off
reg dummy_d_14;
// synthesis translate_on
always @(*) begin
	liteethmac_sel0 <= 2'd0;
	if ((mac_crossbar_sink_payload_ethernet_type == 12'd2054)) begin
		liteethmac_sel0 <= 1'd1;
	end
	if ((mac_crossbar_sink_payload_ethernet_type == 12'd2048)) begin
		liteethmac_sel0 <= 2'd2;
	end
// synthesis translate_off
	dummy_d_14 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_15;
// synthesis translate_on
always @(*) begin
	liteethmac_request <= 2'd0;
	liteethmac_request[0] <= liteethmac_status0_ongoing0;
	liteethmac_request[1] <= liteethmac_status1_ongoing0;
// synthesis translate_off
	dummy_d_15 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_16;
// synthesis translate_on
always @(*) begin
	mac_crossbar_source_valid <= 1'd0;
	mac_crossbar_source_first <= 1'd0;
	mac_crossbar_source_last <= 1'd0;
	mac_crossbar_source_payload_ethernet_type <= 16'd0;
	mac_crossbar_source_payload_sender_mac <= 48'd0;
	mac_crossbar_source_payload_target_mac <= 48'd0;
	mac_crossbar_source_payload_data <= 8'd0;
	mac_crossbar_source_payload_be <= 1'd0;
	mac_crossbar_source_payload_error <= 1'd0;
	arp_mac_port_sink_ready <= 1'd0;
	ip_mac_port_sink_ready <= 1'd0;
	case (liteethmac_grant)
		1'd0: begin
			mac_crossbar_source_valid <= arp_mac_port_sink_valid;
			arp_mac_port_sink_ready <= mac_crossbar_source_ready;
			mac_crossbar_source_first <= arp_mac_port_sink_first;
			mac_crossbar_source_last <= arp_mac_port_sink_last;
			mac_crossbar_source_payload_ethernet_type <= arp_mac_port_sink_payload_ethernet_type;
			mac_crossbar_source_payload_sender_mac <= arp_mac_port_sink_payload_sender_mac;
			mac_crossbar_source_payload_target_mac <= arp_mac_port_sink_payload_target_mac;
			mac_crossbar_source_payload_data <= arp_mac_port_sink_payload_data;
			mac_crossbar_source_payload_be <= arp_mac_port_sink_payload_be;
			mac_crossbar_source_payload_error <= arp_mac_port_sink_payload_error;
		end
		1'd1: begin
			mac_crossbar_source_valid <= ip_mac_port_sink_valid;
			ip_mac_port_sink_ready <= mac_crossbar_source_ready;
			mac_crossbar_source_first <= ip_mac_port_sink_first;
			mac_crossbar_source_last <= ip_mac_port_sink_last;
			mac_crossbar_source_payload_ethernet_type <= ip_mac_port_sink_payload_ethernet_type;
			mac_crossbar_source_payload_sender_mac <= ip_mac_port_sink_payload_sender_mac;
			mac_crossbar_source_payload_target_mac <= ip_mac_port_sink_payload_target_mac;
			mac_crossbar_source_payload_data <= ip_mac_port_sink_payload_data;
			mac_crossbar_source_payload_be <= ip_mac_port_sink_payload_be;
			mac_crossbar_source_payload_error <= ip_mac_port_sink_payload_error;
		end
	endcase
// synthesis translate_off
	dummy_d_16 <= dummy_s;
// synthesis translate_on
end
assign liteethmac_status0_last = ((arp_mac_port_sink_valid & arp_mac_port_sink_last) & arp_mac_port_sink_ready);
assign liteethmac_status0_ongoing0 = ((arp_mac_port_sink_valid | liteethmac_status0_ongoing1) & (~liteethmac_status0_last));
assign liteethmac_status1_last = ((ip_mac_port_sink_valid & ip_mac_port_sink_last) & ip_mac_port_sink_ready);
assign liteethmac_status1_ongoing0 = ((ip_mac_port_sink_valid | liteethmac_status1_ongoing1) & (~liteethmac_status1_last));

// synthesis translate_off
reg dummy_d_17;
// synthesis translate_on
always @(*) begin
	liteethmac_sel1 <= 2'd0;
	if ((~liteethmac_sel_locked)) begin
		liteethmac_sel1 <= liteethmac_sel0;
	end else begin
		liteethmac_sel1 <= liteethmac_sel_ongoing;
	end
// synthesis translate_off
	dummy_d_17 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_18;
// synthesis translate_on
always @(*) begin
	mac_crossbar_sink_ready <= 1'd0;
	arp_mac_port_source_valid <= 1'd0;
	arp_mac_port_source_first <= 1'd0;
	arp_mac_port_source_last <= 1'd0;
	arp_mac_port_source_payload_ethernet_type <= 16'd0;
	arp_mac_port_source_payload_sender_mac <= 48'd0;
	arp_mac_port_source_payload_target_mac <= 48'd0;
	arp_mac_port_source_payload_data <= 8'd0;
	arp_mac_port_source_payload_be <= 1'd0;
	arp_mac_port_source_payload_error <= 1'd0;
	ip_mac_port_source_valid <= 1'd0;
	ip_mac_port_source_first <= 1'd0;
	ip_mac_port_source_last <= 1'd0;
	ip_mac_port_source_payload_ethernet_type <= 16'd0;
	ip_mac_port_source_payload_sender_mac <= 48'd0;
	ip_mac_port_source_payload_target_mac <= 48'd0;
	ip_mac_port_source_payload_data <= 8'd0;
	ip_mac_port_source_payload_be <= 1'd0;
	ip_mac_port_source_payload_error <= 1'd0;
	case (liteethmac_sel1)
		1'd1: begin
			arp_mac_port_source_valid <= mac_crossbar_sink_valid;
			mac_crossbar_sink_ready <= arp_mac_port_source_ready;
			arp_mac_port_source_first <= mac_crossbar_sink_first;
			arp_mac_port_source_last <= mac_crossbar_sink_last;
			arp_mac_port_source_payload_ethernet_type <= mac_crossbar_sink_payload_ethernet_type;
			arp_mac_port_source_payload_sender_mac <= mac_crossbar_sink_payload_sender_mac;
			arp_mac_port_source_payload_target_mac <= mac_crossbar_sink_payload_target_mac;
			arp_mac_port_source_payload_data <= mac_crossbar_sink_payload_data;
			arp_mac_port_source_payload_be <= mac_crossbar_sink_payload_be;
			arp_mac_port_source_payload_error <= mac_crossbar_sink_payload_error;
		end
		2'd2: begin
			ip_mac_port_source_valid <= mac_crossbar_sink_valid;
			mac_crossbar_sink_ready <= ip_mac_port_source_ready;
			ip_mac_port_source_first <= mac_crossbar_sink_first;
			ip_mac_port_source_last <= mac_crossbar_sink_last;
			ip_mac_port_source_payload_ethernet_type <= mac_crossbar_sink_payload_ethernet_type;
			ip_mac_port_source_payload_sender_mac <= mac_crossbar_sink_payload_sender_mac;
			ip_mac_port_source_payload_target_mac <= mac_crossbar_sink_payload_target_mac;
			ip_mac_port_source_payload_data <= mac_crossbar_sink_payload_data;
			ip_mac_port_source_payload_be <= mac_crossbar_sink_payload_be;
			ip_mac_port_source_payload_error <= mac_crossbar_sink_payload_error;
		end
		default: begin
			mac_crossbar_sink_ready <= 1'd1;
		end
	endcase
// synthesis translate_off
	dummy_d_18 <= dummy_s;
// synthesis translate_on
end
assign liteethmac_last = ((mac_crossbar_sink_valid & mac_crossbar_sink_last) & mac_crossbar_sink_ready);
assign liteethmac_ongoing0 = ((mac_crossbar_sink_valid | liteethmac_ongoing1) & (~liteethmac_last));

// synthesis translate_off
reg dummy_d_19;
// synthesis translate_on
always @(*) begin
	mac_packetizer_header <= 112'd0;
	mac_packetizer_header[111:96] <= {mac_packetizer_sink_payload_ethernet_type[7:0], mac_packetizer_sink_payload_ethernet_type[15:8]};
	mac_packetizer_header[95:48] <= {mac_packetizer_sink_payload_sender_mac[7:0], mac_packetizer_sink_payload_sender_mac[15:8], mac_packetizer_sink_payload_sender_mac[23:16], mac_packetizer_sink_payload_sender_mac[31:24], mac_packetizer_sink_payload_sender_mac[39:32], mac_packetizer_sink_payload_sender_mac[47:40]};
	mac_packetizer_header[47:0] <= {mac_packetizer_sink_payload_target_mac[7:0], mac_packetizer_sink_payload_target_mac[15:8], mac_packetizer_sink_payload_target_mac[23:16], mac_packetizer_sink_payload_target_mac[31:24], mac_packetizer_sink_payload_target_mac[39:32], mac_packetizer_sink_payload_target_mac[47:40]};
// synthesis translate_off
	dummy_d_19 <= dummy_s;
// synthesis translate_on
end
assign mac_packetizer_source_payload_be = mac_packetizer_source_be;

// synthesis translate_off
reg dummy_d_20;
// synthesis translate_on
always @(*) begin
	mac_packetizer_source_payload_error <= 1'd0;
	if (mac_packetizer_is_ongoing) begin
		mac_packetizer_source_payload_error <= mac_packetizer_sink_payload_error;
	end
// synthesis translate_off
	dummy_d_20 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_21;
// synthesis translate_on
always @(*) begin
	mac_packetizer_sink_ready <= 1'd0;
	mac_packetizer_source_valid <= 1'd0;
	mac_packetizer_source_first <= 1'd0;
	mac_packetizer_source_last <= 1'd0;
	mac_packetizer_source_payload_data <= 8'd0;
	mac_packetizer_source_be <= 1'd1;
	mac_packetizer_is_ongoing <= 1'd0;
	liteethmac_liteethmacpacketizer_next_state <= 1'd0;
	mac_packetizer_count_liteethmacpacketizer_next_value <= 4'd0;
	mac_packetizer_count_liteethmacpacketizer_next_value_ce <= 1'd0;
	liteethmac_liteethmacpacketizer_next_state <= liteethmac_liteethmacpacketizer_state;
	case (liteethmac_liteethmacpacketizer_state)
		1'd0: begin
			mac_packetizer_is_ongoing <= 1'd1;
			mac_packetizer_source_valid <= mac_packetizer_sink_valid;
			mac_packetizer_sink_ready <= mac_packetizer_source_ready;
			mac_packetizer_source_last <= mac_packetizer_sink_last;
			mac_packetizer_source_payload_data <= mac_packetizer_sink_payload_data;
			mac_packetizer_source_be <= mac_packetizer_sink_payload_be;
			if (((mac_packetizer_source_valid & mac_packetizer_source_ready) & mac_packetizer_source_last)) begin
				liteethmac_liteethmacpacketizer_next_state <= 1'd1;
			end
		end
		default: begin
			mac_packetizer_sink_ready <= (mac_packetizer_count == 1'd0);
			case (mac_packetizer_count)
				1'd0: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[7:0];
				end
				1'd1: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[15:8];
				end
				2'd2: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[23:16];
				end
				2'd3: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[31:24];
				end
				3'd4: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[39:32];
				end
				3'd5: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[47:40];
				end
				3'd6: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[55:48];
				end
				3'd7: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[63:56];
				end
				4'd8: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[71:64];
				end
				4'd9: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[79:72];
				end
				4'd10: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[87:80];
				end
				4'd11: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[95:88];
				end
				4'd12: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[103:96];
				end
				4'd13: begin
					mac_packetizer_source_payload_data <= mac_packetizer_header[111:104];
				end
			endcase
			if ((mac_packetizer_count == 4'd13)) begin
				mac_packetizer_source_be <= 1'd1;
			end
			if ((mac_packetizer_sink_valid & 1'd1)) begin
				mac_packetizer_source_valid <= 1'd1;
				mac_packetizer_source_first <= (mac_packetizer_count == 1'd0);
				mac_packetizer_source_last <= ((mac_packetizer_count == 4'd13) & (mac_packetizer_sink_last & (mac_packetizer_sink_payload_be == 1'd0)));
				mac_packetizer_sink_ready <= (((mac_packetizer_count == 4'd13) & (mac_packetizer_sink_last & (mac_packetizer_sink_payload_be == 1'd0))) & mac_packetizer_source_ready);
				if (mac_packetizer_source_ready) begin
					if ((mac_packetizer_count == 4'd13)) begin
						mac_packetizer_count_liteethmacpacketizer_next_value <= 1'd0;
						mac_packetizer_count_liteethmacpacketizer_next_value_ce <= 1'd1;
						if ((~mac_packetizer_source_last)) begin
							liteethmac_liteethmacpacketizer_next_state <= 1'd0;
						end
					end else begin
						mac_packetizer_count_liteethmacpacketizer_next_value <= (mac_packetizer_count + 1'd1);
						mac_packetizer_count_liteethmacpacketizer_next_value_ce <= 1'd1;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_21 <= dummy_s;
// synthesis translate_on
end
assign mac_depacketizer_header = mac_depacketizer_sr;
assign mac_depacketizer_source_payload_ethernet_type = {rhs_slice_proxy1[7:0], rhs_slice_proxy0[15:8]};
assign mac_depacketizer_source_payload_sender_mac = {rhs_slice_proxy7[7:0], rhs_slice_proxy6[15:8], rhs_slice_proxy5[23:16], rhs_slice_proxy4[31:24], rhs_slice_proxy3[39:32], rhs_slice_proxy2[47:40]};
assign mac_depacketizer_source_payload_target_mac = {rhs_slice_proxy13[7:0], rhs_slice_proxy12[15:8], rhs_slice_proxy11[23:16], rhs_slice_proxy10[31:24], rhs_slice_proxy9[39:32], rhs_slice_proxy8[47:40]};
assign mac_depacketizer_source_payload_error = mac_depacketizer_sink_payload_error;
assign mac_depacketizer_sr_shift = ((((mac_depacketizer_is_ongoing0 | mac_depacketizer_is_ongoing1) & mac_depacketizer_sink_valid) & mac_depacketizer_sink_ready) & (mac_depacketizer_skip_cnt >= 1'd0));

// synthesis translate_off
reg dummy_d_22;
// synthesis translate_on
always @(*) begin
	mac_depacketizer_sink_ready <= 1'd0;
	mac_depacketizer_source_valid <= 1'd0;
	mac_depacketizer_source_first <= 1'd0;
	mac_depacketizer_source_last <= 1'd0;
	mac_depacketizer_source_payload_data <= 8'd0;
	mac_depacketizer_source_payload_be <= 1'd0;
	mac_depacketizer_is_ongoing0 <= 1'd0;
	mac_depacketizer_is_ongoing1 <= 1'd0;
	liteethmac_liteethmacdepacketizer_next_state <= 2'd0;
	mac_depacketizer_count_liteethmacdepacketizer_next_value <= 4'd0;
	mac_depacketizer_count_liteethmacdepacketizer_next_value_ce <= 1'd0;
	mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value0 <= 1'd0;
	mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value_ce0 <= 1'd0;
	mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value1 <= 1'd0;
	mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value_ce1 <= 1'd0;
	mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value <= 1'd0;
	mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value_ce <= 1'd0;
	mac_depacketizer_source_payload_data <= mac_depacketizer_sink_payload_data;
	mac_depacketizer_source_payload_be <= mac_depacketizer_sink_payload_be;
	liteethmac_liteethmacdepacketizer_next_state <= liteethmac_liteethmacdepacketizer_state;
	case (liteethmac_liteethmacdepacketizer_state)
		1'd1: begin
			mac_depacketizer_is_ongoing1 <= 1'd1;
			mac_depacketizer_sink_ready <= 1'd1;
			if ((mac_depacketizer_sink_valid & mac_depacketizer_sink_ready)) begin
				mac_depacketizer_count_liteethmacdepacketizer_next_value <= (mac_depacketizer_count + 1'd1);
				mac_depacketizer_count_liteethmacdepacketizer_next_value_ce <= 1'd1;
				if ((mac_depacketizer_count == 4'd13)) begin
					mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value0 <= 1'd1;
					mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value_ce0 <= 1'd1;
					liteethmac_liteethmacdepacketizer_next_state <= 2'd2;
					mac_depacketizer_count_liteethmacdepacketizer_next_value <= (mac_depacketizer_count + 1'd1);
					mac_depacketizer_count_liteethmacdepacketizer_next_value_ce <= 1'd1;
				end
				if ((mac_depacketizer_sink_last & mac_depacketizer_sink_ready)) begin
					liteethmac_liteethmacdepacketizer_next_state <= 1'd0;
				end
			end
		end
		2'd2: begin
			if ((1'd0 & mac_depacketizer_send_dummy)) begin
				mac_depacketizer_source_valid <= 1'd1;
				mac_depacketizer_source_payload_data <= 1'd0;
				mac_depacketizer_source_first <= 1'd1;
				mac_depacketizer_source_last <= 1'd0;
				mac_depacketizer_source_payload_be <= 1'd1;
				mac_depacketizer_sink_ready <= 1'd0;
				if (mac_depacketizer_source_ready) begin
					mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value1 <= 1'd0;
					mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value_ce1 <= 1'd1;
				end
			end else begin
				mac_depacketizer_source_valid <= mac_depacketizer_sink_valid;
				mac_depacketizer_source_last <= mac_depacketizer_sink_last;
				mac_depacketizer_source_first <= (mac_depacketizer_data_copy_first & mac_depacketizer_sink_valid);
				mac_depacketizer_sink_ready <= mac_depacketizer_source_ready;
				if ((mac_depacketizer_source_valid & mac_depacketizer_source_ready)) begin
					mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value0 <= 1'd0;
					mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value_ce0 <= 1'd1;
					if (mac_depacketizer_source_last) begin
						mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value <= 1'd0;
						mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value_ce <= 1'd1;
						if (1'd0) begin
							mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value1 <= 1'd1;
							mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value_ce1 <= 1'd1;
						end
						liteethmac_liteethmacdepacketizer_next_state <= 1'd0;
					end
				end
			end
		end
		default: begin
			mac_depacketizer_is_ongoing0 <= 1'd1;
			mac_depacketizer_sink_ready <= 1'd1;
			mac_depacketizer_count_liteethmacdepacketizer_next_value <= 1'd1;
			mac_depacketizer_count_liteethmacdepacketizer_next_value_ce <= 1'd1;
			if ((mac_depacketizer_sink_valid & mac_depacketizer_sink_ready)) begin
				liteethmac_liteethmacdepacketizer_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_22 <= dummy_s;
// synthesis translate_on
end
assign arp_table_sink_valid = arp_rx_source_source_valid;
assign arp_rx_source_source_ready = arp_table_sink_ready;
assign arp_table_sink_first = arp_rx_source_source_first;
assign arp_table_sink_last = arp_rx_source_source_last;
assign arp_table_sink_payload_reply = arp_rx_source_source_payload_reply;
assign arp_table_sink_payload_request = arp_rx_source_source_payload_request;
assign arp_table_sink_payload_ip_address = arp_rx_source_source_payload_ip_address;
assign arp_table_sink_payload_mac_address = arp_rx_source_source_payload_mac_address;
assign arp_tx_sink_sink_valid = arp_table_source_valid;
assign arp_table_source_ready = arp_tx_sink_sink_ready;
assign arp_tx_sink_sink_first = arp_table_source_first;
assign arp_tx_sink_sink_last = arp_table_source_last;
assign arp_tx_sink_sink_payload_reply = arp_table_source_payload_reply;
assign arp_tx_sink_sink_payload_request = arp_table_source_payload_request;
assign arp_tx_sink_sink_payload_ip_address = arp_table_source_payload_ip_address;
assign arp_tx_sink_sink_payload_mac_address = arp_table_source_payload_mac_address;
assign arp_mac_port_sink_valid = arp_tx_source_source_valid;
assign arp_tx_source_source_ready = arp_mac_port_sink_ready;
assign arp_mac_port_sink_first = arp_tx_source_source_first;
assign arp_mac_port_sink_last = arp_tx_source_source_last;
assign arp_mac_port_sink_payload_ethernet_type = arp_tx_source_source_payload_ethernet_type;
assign arp_mac_port_sink_payload_sender_mac = arp_tx_source_source_payload_sender_mac;
assign arp_mac_port_sink_payload_target_mac = arp_tx_source_source_payload_target_mac;
assign arp_mac_port_sink_payload_data = arp_tx_source_source_payload_data;
assign arp_mac_port_sink_payload_be = arp_tx_source_source_payload_be;
assign arp_mac_port_sink_payload_error = arp_tx_source_source_payload_error;
assign arp_rx_sink_sink_valid = arp_mac_port_source_valid;
assign arp_mac_port_source_ready = arp_rx_sink_sink_ready;
assign arp_rx_sink_sink_first = arp_mac_port_source_first;
assign arp_rx_sink_sink_last = arp_mac_port_source_last;
assign arp_rx_sink_sink_payload_ethernet_type = arp_mac_port_source_payload_ethernet_type;
assign arp_rx_sink_sink_payload_sender_mac = arp_mac_port_source_payload_sender_mac;
assign arp_rx_sink_sink_payload_target_mac = arp_mac_port_source_payload_target_mac;
assign arp_rx_sink_sink_payload_data = arp_mac_port_source_payload_data;
assign arp_rx_sink_sink_payload_be = arp_mac_port_source_payload_be;
assign arp_rx_sink_sink_payload_error = arp_mac_port_source_payload_error;
assign arp_tx_packetizer_sink_last = (arp_tx_counter == 5'd17);
assign arp_tx_packetizer_sink_payload_be = (arp_tx_packetizer_sink_last ? 1'd1 : 1'd1);
assign arp_tx_packetizer_sink_param_hwtype = 1'd1;
assign arp_tx_packetizer_sink_param_proto = 12'd2048;
assign arp_tx_packetizer_sink_param_hwsize = 3'd6;
assign arp_tx_packetizer_sink_param_protosize = 3'd4;
assign arp_tx_packetizer_sink_param_sender_mac = 42'd2943904740353;
assign arp_tx_packetizer_sink_param_sender_ip = 32'd3232246274;
assign arp_tx_packetizer_sink_param_target_ip = arp_tx_sink_sink_payload_ip_address;

// synthesis translate_off
reg dummy_d_23;
// synthesis translate_on
always @(*) begin
	arp_tx_packetizer_sink_param_opcode <= 16'd0;
	arp_tx_packetizer_sink_param_target_mac <= 48'd0;
	if (arp_tx_sink_sink_payload_reply) begin
		arp_tx_packetizer_sink_param_opcode <= 2'd2;
		arp_tx_packetizer_sink_param_target_mac <= arp_tx_sink_sink_payload_mac_address;
	end else begin
		if (arp_tx_sink_sink_payload_request) begin
			arp_tx_packetizer_sink_param_opcode <= 1'd1;
			arp_tx_packetizer_sink_param_target_mac <= 48'd281474976710655;
		end
	end
// synthesis translate_off
	dummy_d_23 <= dummy_s;
// synthesis translate_on
end
assign arp_tx_source_source_first = arp_tx_packetizer_source_first;
assign arp_tx_source_source_last = arp_tx_packetizer_source_last;
assign arp_tx_source_source_payload_data = arp_tx_packetizer_source_payload_data;
assign arp_tx_source_source_payload_be = arp_tx_packetizer_source_payload_be;
assign arp_tx_source_source_payload_error = arp_tx_packetizer_source_payload_error;

// synthesis translate_off
reg dummy_d_24;
// synthesis translate_on
always @(*) begin
	arp_tx_source_source_payload_target_mac <= 48'd0;
	arp_tx_source_source_payload_target_mac <= arp_tx_packetizer_source_payload_target_mac;
	arp_tx_source_source_payload_target_mac <= arp_tx_packetizer_sink_param_target_mac;
// synthesis translate_off
	dummy_d_24 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_25;
// synthesis translate_on
always @(*) begin
	arp_tx_source_source_payload_sender_mac <= 48'd0;
	arp_tx_source_source_payload_sender_mac <= arp_tx_packetizer_source_payload_sender_mac;
	arp_tx_source_source_payload_sender_mac <= 42'd2943904740353;
// synthesis translate_off
	dummy_d_25 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_26;
// synthesis translate_on
always @(*) begin
	arp_tx_source_source_payload_ethernet_type <= 16'd0;
	arp_tx_source_source_payload_ethernet_type <= arp_tx_packetizer_source_payload_ethernet_type;
	arp_tx_source_source_payload_ethernet_type <= 12'd2054;
// synthesis translate_off
	dummy_d_26 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_27;
// synthesis translate_on
always @(*) begin
	arp_tx_packetizer_header <= 224'd0;
	arp_tx_packetizer_header[39:32] <= {arp_tx_packetizer_sink_param_hwsize[7:0]};
	arp_tx_packetizer_header[15:0] <= {arp_tx_packetizer_sink_param_hwtype[7:0], arp_tx_packetizer_sink_param_hwtype[15:8]};
	arp_tx_packetizer_header[63:48] <= {arp_tx_packetizer_sink_param_opcode[7:0], arp_tx_packetizer_sink_param_opcode[15:8]};
	arp_tx_packetizer_header[31:16] <= {arp_tx_packetizer_sink_param_proto[7:0], arp_tx_packetizer_sink_param_proto[15:8]};
	arp_tx_packetizer_header[47:40] <= {arp_tx_packetizer_sink_param_protosize[7:0]};
	arp_tx_packetizer_header[143:112] <= {arp_tx_packetizer_sink_param_sender_ip[7:0], arp_tx_packetizer_sink_param_sender_ip[15:8], arp_tx_packetizer_sink_param_sender_ip[23:16], arp_tx_packetizer_sink_param_sender_ip[31:24]};
	arp_tx_packetizer_header[111:64] <= {arp_tx_packetizer_sink_param_sender_mac[7:0], arp_tx_packetizer_sink_param_sender_mac[15:8], arp_tx_packetizer_sink_param_sender_mac[23:16], arp_tx_packetizer_sink_param_sender_mac[31:24], arp_tx_packetizer_sink_param_sender_mac[39:32], arp_tx_packetizer_sink_param_sender_mac[47:40]};
	arp_tx_packetizer_header[223:192] <= {arp_tx_packetizer_sink_param_target_ip[7:0], arp_tx_packetizer_sink_param_target_ip[15:8], arp_tx_packetizer_sink_param_target_ip[23:16], arp_tx_packetizer_sink_param_target_ip[31:24]};
	arp_tx_packetizer_header[191:144] <= {arp_tx_packetizer_sink_param_target_mac[7:0], arp_tx_packetizer_sink_param_target_mac[15:8], arp_tx_packetizer_sink_param_target_mac[23:16], arp_tx_packetizer_sink_param_target_mac[31:24], arp_tx_packetizer_sink_param_target_mac[39:32], arp_tx_packetizer_sink_param_target_mac[47:40]};
// synthesis translate_off
	dummy_d_27 <= dummy_s;
// synthesis translate_on
end
assign arp_tx_packetizer_source_payload_be = arp_tx_packetizer_source_be;

// synthesis translate_off
reg dummy_d_28;
// synthesis translate_on
always @(*) begin
	arp_tx_packetizer_source_payload_error <= 1'd0;
	if (arp_tx_packetizer_is_ongoing) begin
		arp_tx_packetizer_source_payload_error <= arp_tx_packetizer_sink_payload_error;
	end
// synthesis translate_off
	dummy_d_28 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_29;
// synthesis translate_on
always @(*) begin
	arp_tx_packetizer_sink_ready <= 1'd0;
	arp_tx_packetizer_source_valid <= 1'd0;
	arp_tx_packetizer_source_first <= 1'd0;
	arp_tx_packetizer_source_last <= 1'd0;
	arp_tx_packetizer_source_payload_data <= 8'd0;
	arp_tx_packetizer_source_be <= 1'd1;
	arp_tx_packetizer_is_ongoing <= 1'd0;
	liteetharptx_liteetharppacketizer_next_state <= 1'd0;
	arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value <= 5'd0;
	arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value_ce <= 1'd0;
	liteetharptx_liteetharppacketizer_next_state <= liteetharptx_liteetharppacketizer_state;
	case (liteetharptx_liteetharppacketizer_state)
		1'd0: begin
			arp_tx_packetizer_is_ongoing <= 1'd1;
			arp_tx_packetizer_source_valid <= arp_tx_packetizer_sink_valid;
			arp_tx_packetizer_sink_ready <= arp_tx_packetizer_source_ready;
			arp_tx_packetizer_source_last <= arp_tx_packetizer_sink_last;
			arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_sink_payload_data;
			arp_tx_packetizer_source_be <= arp_tx_packetizer_sink_payload_be;
			if (((arp_tx_packetizer_source_valid & arp_tx_packetizer_source_ready) & arp_tx_packetizer_source_last)) begin
				liteetharptx_liteetharppacketizer_next_state <= 1'd1;
			end
		end
		default: begin
			arp_tx_packetizer_sink_ready <= (arp_tx_packetizer_count == 1'd0);
			case (arp_tx_packetizer_count)
				1'd0: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[7:0];
				end
				1'd1: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[15:8];
				end
				2'd2: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[23:16];
				end
				2'd3: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[31:24];
				end
				3'd4: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[39:32];
				end
				3'd5: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[47:40];
				end
				3'd6: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[55:48];
				end
				3'd7: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[63:56];
				end
				4'd8: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[71:64];
				end
				4'd9: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[79:72];
				end
				4'd10: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[87:80];
				end
				4'd11: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[95:88];
				end
				4'd12: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[103:96];
				end
				4'd13: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[111:104];
				end
				4'd14: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[119:112];
				end
				4'd15: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[127:120];
				end
				5'd16: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[135:128];
				end
				5'd17: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[143:136];
				end
				5'd18: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[151:144];
				end
				5'd19: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[159:152];
				end
				5'd20: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[167:160];
				end
				5'd21: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[175:168];
				end
				5'd22: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[183:176];
				end
				5'd23: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[191:184];
				end
				5'd24: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[199:192];
				end
				5'd25: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[207:200];
				end
				5'd26: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[215:208];
				end
				5'd27: begin
					arp_tx_packetizer_source_payload_data <= arp_tx_packetizer_header[223:216];
				end
			endcase
			if ((arp_tx_packetizer_count == 5'd27)) begin
				arp_tx_packetizer_source_be <= 1'd1;
			end
			if ((arp_tx_packetizer_sink_valid & 1'd1)) begin
				arp_tx_packetizer_source_valid <= 1'd1;
				arp_tx_packetizer_source_first <= (arp_tx_packetizer_count == 1'd0);
				arp_tx_packetizer_source_last <= ((arp_tx_packetizer_count == 5'd27) & (arp_tx_packetizer_sink_last & (arp_tx_packetizer_sink_payload_be == 1'd0)));
				arp_tx_packetizer_sink_ready <= (((arp_tx_packetizer_count == 5'd27) & (arp_tx_packetizer_sink_last & (arp_tx_packetizer_sink_payload_be == 1'd0))) & arp_tx_packetizer_source_ready);
				if (arp_tx_packetizer_source_ready) begin
					if ((arp_tx_packetizer_count == 5'd27)) begin
						arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value <= 1'd0;
						arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value_ce <= 1'd1;
						if ((~arp_tx_packetizer_source_last)) begin
							liteetharptx_liteetharppacketizer_next_state <= 1'd0;
						end
					end else begin
						arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value <= (arp_tx_packetizer_count + 1'd1);
						arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value_ce <= 1'd1;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_29 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_30;
// synthesis translate_on
always @(*) begin
	arp_tx_sink_sink_ready <= 1'd0;
	arp_tx_source_source_valid <= 1'd0;
	arp_tx_packetizer_sink_valid <= 1'd0;
	arp_tx_packetizer_source_ready <= 1'd0;
	liteetharptx_fsm_next_state <= 1'd0;
	arp_tx_counter_liteetharp_liteetharptx_fsm_next_value <= 5'd0;
	arp_tx_counter_liteetharp_liteetharptx_fsm_next_value_ce <= 1'd0;
	liteetharptx_fsm_next_state <= liteetharptx_fsm_state;
	case (liteetharptx_fsm_state)
		1'd1: begin
			arp_tx_packetizer_sink_valid <= 1'd1;
			arp_tx_source_source_valid <= arp_tx_packetizer_source_valid;
			arp_tx_packetizer_source_ready <= arp_tx_source_source_ready;
			if ((arp_tx_packetizer_sink_valid & arp_tx_packetizer_sink_ready)) begin
				arp_tx_counter_liteetharp_liteetharptx_fsm_next_value <= (arp_tx_counter + 1'd1);
				arp_tx_counter_liteetharp_liteetharptx_fsm_next_value_ce <= 1'd1;
			end
			if ((arp_tx_source_source_valid & arp_tx_source_source_ready)) begin
				if (arp_tx_source_source_last) begin
					arp_tx_sink_sink_ready <= 1'd1;
					liteetharptx_fsm_next_state <= 1'd0;
				end
			end
		end
		default: begin
			arp_tx_counter_liteetharp_liteetharptx_fsm_next_value <= 1'd0;
			arp_tx_counter_liteetharp_liteetharptx_fsm_next_value_ce <= 1'd1;
			if (arp_tx_sink_sink_valid) begin
				liteetharptx_fsm_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_30 <= dummy_s;
// synthesis translate_on
end
assign arp_rx_depacketizer_sink_valid = arp_rx_sink_sink_valid;
assign arp_rx_sink_sink_ready = arp_rx_depacketizer_sink_ready;
assign arp_rx_depacketizer_sink_first = arp_rx_sink_sink_first;
assign arp_rx_depacketizer_sink_last = arp_rx_sink_sink_last;
assign arp_rx_depacketizer_sink_payload_ethernet_type = arp_rx_sink_sink_payload_ethernet_type;
assign arp_rx_depacketizer_sink_payload_sender_mac = arp_rx_sink_sink_payload_sender_mac;
assign arp_rx_depacketizer_sink_payload_target_mac = arp_rx_sink_sink_payload_target_mac;
assign arp_rx_depacketizer_sink_payload_data = arp_rx_sink_sink_payload_data;
assign arp_rx_depacketizer_sink_payload_be = arp_rx_sink_sink_payload_be;
assign arp_rx_depacketizer_sink_payload_error = arp_rx_sink_sink_payload_error;

// synthesis translate_off
reg dummy_d_31;
// synthesis translate_on
always @(*) begin
	arp_rx_reply <= 1'd0;
	arp_rx_request <= 1'd0;
	case (arp_rx_depacketizer_source_param_opcode)
		1'd1: begin
			arp_rx_request <= 1'd1;
		end
		2'd2: begin
			arp_rx_reply <= 1'd1;
		end
		default: begin
		end
	endcase
// synthesis translate_off
	dummy_d_31 <= dummy_s;
// synthesis translate_on
end
assign arp_rx_source_source_payload_ip_address = arp_rx_depacketizer_source_param_sender_ip;
assign arp_rx_source_source_payload_mac_address = arp_rx_depacketizer_source_param_sender_mac;
assign arp_rx_depacketizer_header = arp_rx_depacketizer_sr;
assign arp_rx_depacketizer_source_param_hwsize = {rhs_slice_proxy14[7:0]};
assign arp_rx_depacketizer_source_param_hwtype = {rhs_slice_proxy16[7:0], rhs_slice_proxy15[15:8]};
assign arp_rx_depacketizer_source_param_opcode = {rhs_slice_proxy18[7:0], rhs_slice_proxy17[15:8]};
assign arp_rx_depacketizer_source_param_proto = {rhs_slice_proxy20[7:0], rhs_slice_proxy19[15:8]};
assign arp_rx_depacketizer_source_param_protosize = {rhs_slice_proxy21[7:0]};
assign arp_rx_depacketizer_source_param_sender_ip = {rhs_slice_proxy25[7:0], rhs_slice_proxy24[15:8], rhs_slice_proxy23[23:16], rhs_slice_proxy22[31:24]};
assign arp_rx_depacketizer_source_param_sender_mac = {rhs_slice_proxy31[7:0], rhs_slice_proxy30[15:8], rhs_slice_proxy29[23:16], rhs_slice_proxy28[31:24], rhs_slice_proxy27[39:32], rhs_slice_proxy26[47:40]};
assign arp_rx_depacketizer_source_param_target_ip = {rhs_slice_proxy35[7:0], rhs_slice_proxy34[15:8], rhs_slice_proxy33[23:16], rhs_slice_proxy32[31:24]};
assign arp_rx_depacketizer_source_param_target_mac = {rhs_slice_proxy41[7:0], rhs_slice_proxy40[15:8], rhs_slice_proxy39[23:16], rhs_slice_proxy38[31:24], rhs_slice_proxy37[39:32], rhs_slice_proxy36[47:40]};
assign arp_rx_depacketizer_source_payload_error = arp_rx_depacketizer_sink_payload_error;
assign arp_rx_depacketizer_sr_shift = ((((arp_rx_depacketizer_is_ongoing0 | arp_rx_depacketizer_is_ongoing1) & arp_rx_depacketizer_sink_valid) & arp_rx_depacketizer_sink_ready) & (arp_rx_depacketizer_skip_cnt >= 1'd0));

// synthesis translate_off
reg dummy_d_32;
// synthesis translate_on
always @(*) begin
	arp_rx_depacketizer_sink_ready <= 1'd0;
	arp_rx_depacketizer_source_valid <= 1'd0;
	arp_rx_depacketizer_source_first <= 1'd0;
	arp_rx_depacketizer_source_last <= 1'd0;
	arp_rx_depacketizer_source_payload_data <= 8'd0;
	arp_rx_depacketizer_source_payload_be <= 1'd0;
	arp_rx_depacketizer_is_ongoing0 <= 1'd0;
	arp_rx_depacketizer_is_ongoing1 <= 1'd0;
	liteetharprx_liteetharpdepacketizer_next_state <= 2'd0;
	arp_rx_depacketizer_count_liteetharp_next_value <= 5'd0;
	arp_rx_depacketizer_count_liteetharp_next_value_ce <= 1'd0;
	arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value0 <= 1'd0;
	arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value_ce0 <= 1'd0;
	arp_rx_depacketizer_send_dummy_liteetharp_t_next_value1 <= 1'd0;
	arp_rx_depacketizer_send_dummy_liteetharp_t_next_value_ce1 <= 1'd0;
	arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value <= 1'd0;
	arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value_ce <= 1'd0;
	arp_rx_depacketizer_source_payload_data <= arp_rx_depacketizer_sink_payload_data;
	arp_rx_depacketizer_source_payload_be <= arp_rx_depacketizer_sink_payload_be;
	liteetharprx_liteetharpdepacketizer_next_state <= liteetharprx_liteetharpdepacketizer_state;
	case (liteetharprx_liteetharpdepacketizer_state)
		1'd1: begin
			arp_rx_depacketizer_is_ongoing1 <= 1'd1;
			arp_rx_depacketizer_sink_ready <= 1'd1;
			if ((arp_rx_depacketizer_sink_valid & arp_rx_depacketizer_sink_ready)) begin
				arp_rx_depacketizer_count_liteetharp_next_value <= (arp_rx_depacketizer_count + 1'd1);
				arp_rx_depacketizer_count_liteetharp_next_value_ce <= 1'd1;
				if ((arp_rx_depacketizer_count == 5'd27)) begin
					arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value0 <= 1'd1;
					arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value_ce0 <= 1'd1;
					liteetharprx_liteetharpdepacketizer_next_state <= 2'd2;
					arp_rx_depacketizer_count_liteetharp_next_value <= (arp_rx_depacketizer_count + 1'd1);
					arp_rx_depacketizer_count_liteetharp_next_value_ce <= 1'd1;
				end
				if ((arp_rx_depacketizer_sink_last & arp_rx_depacketizer_sink_ready)) begin
					liteetharprx_liteetharpdepacketizer_next_state <= 1'd0;
				end
			end
		end
		2'd2: begin
			if ((1'd0 & arp_rx_depacketizer_send_dummy)) begin
				arp_rx_depacketizer_source_valid <= 1'd1;
				arp_rx_depacketizer_source_payload_data <= 1'd0;
				arp_rx_depacketizer_source_first <= 1'd1;
				arp_rx_depacketizer_source_last <= 1'd0;
				arp_rx_depacketizer_source_payload_be <= 1'd1;
				arp_rx_depacketizer_sink_ready <= 1'd0;
				if (arp_rx_depacketizer_source_ready) begin
					arp_rx_depacketizer_send_dummy_liteetharp_t_next_value1 <= 1'd0;
					arp_rx_depacketizer_send_dummy_liteetharp_t_next_value_ce1 <= 1'd1;
				end
			end else begin
				arp_rx_depacketizer_source_valid <= arp_rx_depacketizer_sink_valid;
				arp_rx_depacketizer_source_last <= arp_rx_depacketizer_sink_last;
				arp_rx_depacketizer_source_first <= (arp_rx_depacketizer_data_copy_first & arp_rx_depacketizer_sink_valid);
				arp_rx_depacketizer_sink_ready <= arp_rx_depacketizer_source_ready;
				if ((arp_rx_depacketizer_source_valid & arp_rx_depacketizer_source_ready)) begin
					arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value0 <= 1'd0;
					arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value_ce0 <= 1'd1;
					if (arp_rx_depacketizer_source_last) begin
						arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value <= 1'd0;
						arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value_ce <= 1'd1;
						if (1'd0) begin
							arp_rx_depacketizer_send_dummy_liteetharp_t_next_value1 <= 1'd1;
							arp_rx_depacketizer_send_dummy_liteetharp_t_next_value_ce1 <= 1'd1;
						end
						liteetharprx_liteetharpdepacketizer_next_state <= 1'd0;
					end
				end
			end
		end
		default: begin
			arp_rx_depacketizer_is_ongoing0 <= 1'd1;
			arp_rx_depacketizer_sink_ready <= 1'd1;
			arp_rx_depacketizer_count_liteetharp_next_value <= 1'd1;
			arp_rx_depacketizer_count_liteetharp_next_value_ce <= 1'd1;
			if ((arp_rx_depacketizer_sink_valid & arp_rx_depacketizer_sink_ready)) begin
				liteetharprx_liteetharpdepacketizer_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_32 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_33;
// synthesis translate_on
always @(*) begin
	arp_rx_source_source_valid <= 1'd0;
	arp_rx_source_source_payload_reply <= 1'd0;
	arp_rx_source_source_payload_request <= 1'd0;
	arp_rx_depacketizer_source_ready <= 1'd0;
	liteetharprx_fsm_next_state <= 2'd0;
	liteetharprx_fsm_next_state <= liteetharprx_fsm_state;
	case (liteetharprx_fsm_state)
		1'd1: begin
			if (arp_rx_valid) begin
				arp_rx_source_source_valid <= 1'd1;
				arp_rx_source_source_payload_reply <= arp_rx_reply;
				arp_rx_source_source_payload_request <= arp_rx_request;
			end
			liteetharprx_fsm_next_state <= 2'd2;
		end
		2'd2: begin
			arp_rx_depacketizer_source_ready <= 1'd1;
			if ((arp_rx_depacketizer_source_valid & arp_rx_depacketizer_source_last)) begin
				liteetharprx_fsm_next_state <= 1'd0;
			end
		end
		default: begin
			arp_rx_depacketizer_source_ready <= 1'd1;
			if (arp_rx_depacketizer_source_valid) begin
				arp_rx_depacketizer_source_ready <= 1'd0;
				liteetharprx_fsm_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_33 <= dummy_s;
// synthesis translate_on
end
assign arp_table_request_timer_wait = (arp_table_request_pending & (~arp_table_request_timer_done));
assign arp_table_request_timer_done = (arp_table_request_timer_count == 1'd0);

// synthesis translate_off
reg dummy_d_34;
// synthesis translate_on
always @(*) begin
	arp_table_cache_mem_wr_port_dat_w <= 81'd0;
	arp_table_cache_mem_wr_port_dat_w[80] <= arp_table_cache_mem_wr_port_valid;
	arp_table_cache_mem_wr_port_dat_w[31:0] <= arp_table_cache_mem_wr_port_ip_address;
	arp_table_cache_mem_wr_port_dat_w[79:32] <= arp_table_cache_mem_wr_port_mac_address;
// synthesis translate_off
	dummy_d_34 <= dummy_s;
// synthesis translate_on
end
assign arp_table_cache_mem_rd_port_valid = arp_table_cache_mem_rd_port_dat_r[80];
assign arp_table_cache_mem_rd_port_ip_address = arp_table_cache_mem_rd_port_dat_r[31:0];
assign arp_table_cache_mem_rd_port_mac_address = arp_table_cache_mem_rd_port_dat_r[79:32];
assign arp_table_cache_done = (arp_table_cache_count == 1'd0);

// synthesis translate_off
reg dummy_d_35;
// synthesis translate_on
always @(*) begin
	arp_table_cache_update_ready <= 1'd0;
	arp_table_cache_request_ready <= 1'd0;
	arp_table_cache_response_valid <= 1'd0;
	arp_table_cache_response_payload_mac_address <= 48'd0;
	arp_table_cache_response_payload_error <= 1'd0;
	arp_table_cache_mem_wr_port_adr <= 1'd0;
	arp_table_cache_mem_wr_port_we <= 1'd0;
	arp_table_cache_mem_rd_port_adr <= 1'd0;
	arp_table_cache_mem_wr_port_valid <= 1'd0;
	arp_table_cache_mem_wr_port_ip_address <= 32'd0;
	arp_table_cache_mem_wr_port_mac_address <= 48'd0;
	arp_table_cache_wait <= 1'd0;
	liteetharptable_liteetharpcache_next_state <= 3'd0;
	arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0 <= 1'd0;
	arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0 <= 1'd0;
	arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value1 <= 1'd0;
	arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce1 <= 1'd0;
	arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value2 <= 1'd0;
	arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value_ce2 <= 1'd0;
	liteetharptable_liteetharpcache_next_state <= liteetharptable_liteetharpcache_state;
	case (liteetharptable_liteetharpcache_state)
		1'd1: begin
			if ((arp_table_cache_enable & arp_table_cache_update_valid)) begin
				liteetharptable_liteetharpcache_next_state <= 2'd2;
			end
			if ((arp_table_cache_enable & arp_table_cache_request_valid)) begin
				arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value1 <= 1'd0;
				arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce1 <= 1'd1;
				liteetharptable_liteetharpcache_next_state <= 2'd3;
			end
			arp_table_cache_wait <= arp_table_cache_clear_enable;
			if (arp_table_cache_done) begin
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0 <= 1'd0;
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0 <= 1'd1;
				liteetharptable_liteetharpcache_next_state <= 1'd0;
			end
		end
		2'd2: begin
			arp_table_cache_mem_wr_port_we <= 1'd1;
			arp_table_cache_mem_wr_port_adr <= arp_table_cache_update_count;
			arp_table_cache_mem_wr_port_valid <= 1'd1;
			arp_table_cache_mem_wr_port_ip_address <= arp_table_cache_update_payload_ip_address;
			arp_table_cache_mem_wr_port_mac_address <= arp_table_cache_update_payload_mac_address;
			arp_table_cache_update_ready <= 1'd1;
			if ((arp_table_cache_update_count == 1'd1)) begin
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0 <= 1'd0;
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0 <= 1'd1;
			end else begin
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0 <= (arp_table_cache_update_count + 1'd1);
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0 <= 1'd1;
			end
			liteetharptable_liteetharpcache_next_state <= 1'd1;
		end
		2'd3: begin
			arp_table_cache_mem_rd_port_adr <= arp_table_cache_search_count;
			if ((arp_table_cache_mem_rd_port_valid & (arp_table_cache_mem_rd_port_ip_address == arp_table_cache_request_payload_ip_address))) begin
				arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value2 <= 1'd0;
				arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value_ce2 <= 1'd1;
				liteetharptable_liteetharpcache_next_state <= 3'd4;
			end else begin
				if ((arp_table_cache_search_count == 1'd1)) begin
					arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value2 <= 1'd1;
					arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value_ce2 <= 1'd1;
					liteetharptable_liteetharpcache_next_state <= 3'd4;
				end else begin
					arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value1 <= (arp_table_cache_search_count + 1'd1);
					arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce1 <= 1'd1;
				end
			end
		end
		3'd4: begin
			arp_table_cache_mem_rd_port_adr <= arp_table_cache_search_count;
			arp_table_cache_request_ready <= 1'd1;
			arp_table_cache_response_valid <= 1'd1;
			arp_table_cache_response_payload_error <= arp_table_cache_error;
			arp_table_cache_response_payload_mac_address <= arp_table_cache_mem_rd_port_mac_address;
			liteetharptable_liteetharpcache_next_state <= 1'd1;
		end
		default: begin
			arp_table_cache_mem_wr_port_we <= 1'd1;
			arp_table_cache_mem_wr_port_adr <= arp_table_cache_update_count;
			arp_table_cache_mem_wr_port_valid <= 1'd0;
			arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0 <= (arp_table_cache_update_count + 1'd1);
			arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0 <= 1'd1;
			if ((arp_table_cache_update_count == 1'd1)) begin
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0 <= 1'd0;
				arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0 <= 1'd1;
				liteetharptable_liteetharpcache_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_35 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_36;
// synthesis translate_on
always @(*) begin
	arp_table_source_valid <= 1'd0;
	arp_table_source_payload_reply <= 1'd0;
	arp_table_source_payload_request <= 1'd0;
	arp_table_source_payload_ip_address <= 32'd0;
	arp_table_source_payload_mac_address <= 48'd0;
	arp_table_request_request_ready <= 1'd0;
	arp_table_response_response_valid <= 1'd0;
	arp_table_cache_update_valid <= 1'd0;
	arp_table_cache_update_payload_ip_address <= 32'd0;
	arp_table_cache_update_payload_mac_address <= 48'd0;
	arp_table_cache_request_valid <= 1'd0;
	arp_table_cache_request_payload_ip_address <= 32'd0;
	liteetharptable_fsm_next_state <= 3'd0;
	arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value0 <= 1'd0;
	arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value_ce0 <= 1'd0;
	arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value1 <= 48'd0;
	arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value_ce1 <= 1'd0;
	arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value2 <= 1'd0;
	arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value_ce2 <= 1'd0;
	arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value3 <= 4'd0;
	arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value_ce3 <= 1'd0;
	arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value4 <= 32'd0;
	arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value_ce4 <= 1'd0;
	liteetharptable_fsm_next_state <= liteetharptable_fsm_state;
	case (liteetharptable_fsm_state)
		1'd1: begin
			arp_table_source_valid <= 1'd1;
			arp_table_source_payload_reply <= 1'd1;
			arp_table_source_payload_ip_address <= arp_table_sink_payload_ip_address;
			arp_table_source_payload_mac_address <= arp_table_sink_payload_mac_address;
			if (arp_table_source_ready) begin
				liteetharptable_fsm_next_state <= 1'd0;
			end
		end
		2'd2: begin
			if ((arp_table_request_pending & (arp_table_request_ip_address == arp_table_sink_payload_ip_address))) begin
				arp_table_cache_update_valid <= 1'd1;
				arp_table_cache_update_payload_ip_address <= arp_table_sink_payload_ip_address;
				arp_table_cache_update_payload_mac_address <= arp_table_sink_payload_mac_address;
				if (arp_table_cache_update_ready) begin
					arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value0 <= 1'd0;
					arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value_ce0 <= 1'd1;
					arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value1 <= arp_table_sink_payload_mac_address;
					arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value_ce1 <= 1'd1;
					liteetharptable_fsm_next_state <= 3'd6;
				end
			end else begin
				liteetharptable_fsm_next_state <= 1'd0;
			end
		end
		2'd3: begin
			if ((arp_table_request_counter == 4'd8)) begin
				arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value2 <= 1'd1;
				arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value_ce2 <= 1'd1;
				arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value3 <= 1'd0;
				arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value_ce3 <= 1'd1;
				arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value0 <= 1'd0;
				arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value_ce0 <= 1'd1;
				liteetharptable_fsm_next_state <= 3'd6;
			end else begin
				liteetharptable_fsm_next_state <= 3'd5;
			end
		end
		3'd4: begin
			arp_table_cache_request_valid <= 1'd1;
			arp_table_cache_request_payload_ip_address <= arp_table_request_request_payload_ip_address;
			if (arp_table_cache_response_valid) begin
				arp_table_request_request_ready <= 1'd1;
				if (arp_table_cache_response_payload_error) begin
					arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value3 <= 1'd0;
					arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value_ce3 <= 1'd1;
					arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value0 <= 1'd1;
					arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value_ce0 <= 1'd1;
					arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value4 <= arp_table_request_request_payload_ip_address;
					arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value_ce4 <= 1'd1;
					liteetharptable_fsm_next_state <= 3'd5;
				end else begin
					arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value1 <= arp_table_cache_response_payload_mac_address;
					arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value_ce1 <= 1'd1;
					liteetharptable_fsm_next_state <= 3'd6;
				end
			end
		end
		3'd5: begin
			arp_table_source_valid <= 1'd1;
			arp_table_source_payload_request <= 1'd1;
			arp_table_source_payload_ip_address <= arp_table_request_ip_address;
			if (arp_table_source_ready) begin
				arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value3 <= (arp_table_request_counter + 1'd1);
				arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value_ce3 <= 1'd1;
				liteetharptable_fsm_next_state <= 1'd0;
			end
		end
		3'd6: begin
			arp_table_response_response_valid <= 1'd1;
			if (arp_table_response_response_ready) begin
				arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value2 <= 1'd0;
				arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value_ce2 <= 1'd1;
				liteetharptable_fsm_next_state <= 1'd0;
			end
		end
		default: begin
			if ((arp_table_sink_valid & arp_table_sink_payload_request)) begin
				liteetharptable_fsm_next_state <= 1'd1;
			end else begin
				if ((arp_table_sink_valid & arp_table_sink_payload_reply)) begin
					liteetharptable_fsm_next_state <= 2'd2;
				end else begin
					if (arp_table_request_request_valid) begin
						liteetharptable_fsm_next_state <= 3'd4;
					end else begin
						if (arp_table_request_timer_done) begin
							liteetharptable_fsm_next_state <= 2'd3;
						end
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_36 <= dummy_s;
// synthesis translate_on
end
assign ip_mac_port_sink_valid = ip_tx_source_source_valid0;
assign ip_tx_source_source_ready0 = ip_mac_port_sink_ready;
assign ip_mac_port_sink_first = ip_tx_source_source_first0;
assign ip_mac_port_sink_last = ip_tx_source_source_last0;
assign ip_mac_port_sink_payload_ethernet_type = ip_tx_source_source_payload_ethernet_type;
assign ip_mac_port_sink_payload_sender_mac = ip_tx_source_source_payload_sender_mac;
assign ip_mac_port_sink_payload_target_mac = ip_tx_source_source_payload_target_mac;
assign ip_mac_port_sink_payload_data = ip_tx_source_source_payload_data0;
assign ip_mac_port_sink_payload_be = ip_tx_source_source_payload_be0;
assign ip_mac_port_sink_payload_error = ip_tx_source_source_payload_error0;
assign ip_rx_sink_sink_valid = ip_mac_port_source_valid;
assign ip_mac_port_source_ready = ip_rx_sink_sink_ready;
assign ip_rx_sink_sink_first = ip_mac_port_source_first;
assign ip_rx_sink_sink_last = ip_mac_port_source_last;
assign ip_rx_sink_sink_payload_ethernet_type = ip_mac_port_source_payload_ethernet_type;
assign ip_rx_sink_sink_payload_sender_mac = ip_mac_port_source_payload_sender_mac;
assign ip_rx_sink_sink_payload_target_mac = ip_mac_port_source_payload_target_mac;
assign ip_rx_sink_sink_payload_data = ip_mac_port_source_payload_data;
assign ip_rx_sink_sink_payload_be = ip_mac_port_source_payload_be;
assign ip_rx_sink_sink_payload_error = ip_mac_port_source_payload_error;
assign ip_tx_sink_sink_valid0 = ip_crossbar_source_valid;
assign ip_crossbar_source_ready = ip_tx_sink_sink_ready0;
assign ip_tx_sink_sink_first0 = ip_crossbar_source_first;
assign ip_tx_sink_sink_last0 = ip_crossbar_source_last;
assign ip_tx_sink_sink_payload_data0 = ip_crossbar_source_payload_data;
assign ip_tx_sink_sink_payload_be0 = ip_crossbar_source_payload_be;
assign ip_tx_sink_sink_payload_error0 = ip_crossbar_source_payload_error;
assign ip_tx_sink_sink_param_length0 = ip_crossbar_source_param_length;
assign ip_tx_sink_sink_param_protocol0 = ip_crossbar_source_param_protocol;
assign ip_tx_sink_sink_param_ip_address0 = ip_crossbar_source_param_ip_address;
assign ip_rx_buffer_sink_sink_valid = ip_rx_source_source_valid;
assign ip_rx_source_source_ready = ip_rx_buffer_sink_sink_ready;
assign ip_rx_buffer_sink_sink_first = ip_rx_source_source_first;
assign ip_rx_buffer_sink_sink_last = ip_rx_source_source_last;
assign ip_rx_buffer_sink_sink_payload_data = ip_rx_source_source_payload_data;
assign ip_rx_buffer_sink_sink_payload_be = ip_rx_source_source_payload_be;
assign ip_rx_buffer_sink_sink_payload_error = ip_rx_source_source_payload_error;
assign ip_rx_buffer_sink_sink_param_length = ip_rx_source_source_param_length;
assign ip_rx_buffer_sink_sink_param_protocol = ip_rx_source_source_param_protocol;
assign ip_rx_buffer_sink_sink_param_ip_address = ip_rx_source_source_param_ip_address;
assign ip_crossbar_sink_valid = ip_rx_buffer_source_source_valid;
assign ip_rx_buffer_source_source_ready = ip_crossbar_sink_ready;
assign ip_crossbar_sink_first = ip_rx_buffer_source_source_first;
assign ip_crossbar_sink_last = ip_rx_buffer_source_source_last;
assign ip_crossbar_sink_payload_data = ip_rx_buffer_source_source_payload_data;
assign ip_crossbar_sink_payload_be = ip_rx_buffer_source_source_payload_be;
assign ip_crossbar_sink_payload_error = ip_rx_buffer_source_source_payload_error;
assign ip_crossbar_sink_param_length = ip_rx_buffer_source_source_param_length;
assign ip_crossbar_sink_param_protocol = ip_rx_buffer_source_source_param_protocol;
assign ip_crossbar_sink_param_ip_address = ip_rx_buffer_source_source_param_ip_address;
assign ip_tx_sink_sink_valid1 = ip_tx_sink_sink_valid0;
assign ip_tx_sink_sink_ready0 = ip_tx_sink_sink_ready1;
assign ip_tx_sink_sink_first1 = ip_tx_sink_sink_first0;
assign ip_tx_sink_sink_last1 = ip_tx_sink_sink_last0;
assign ip_tx_sink_sink_payload_data1 = ip_tx_sink_sink_payload_data0;
assign ip_tx_sink_sink_payload_be1 = ip_tx_sink_sink_payload_be0;
assign ip_tx_sink_sink_payload_error1 = ip_tx_sink_sink_payload_error0;
assign ip_tx_sink_sink_param_length1 = ip_tx_sink_sink_param_length0;
assign ip_tx_sink_sink_param_protocol1 = ip_tx_sink_sink_param_protocol0;
assign ip_tx_sink_sink_param_ip_address1 = ip_tx_sink_sink_param_ip_address0;
assign ip_tx_ce = ip_tx_source_source_valid1;
assign ip_tx_reset = ((ip_tx_packetizer_source_valid & ip_tx_packetizer_source_last) & ip_tx_packetizer_source_ready);
assign ip_tx_packetizer_sink_last = ip_tx_source_source_last1;
assign ip_tx_packetizer_sink_payload_data = ip_tx_source_source_payload_data1;
assign ip_tx_packetizer_sink_payload_be = ip_tx_source_source_payload_be1;
assign ip_tx_packetizer_sink_param_protocol = ip_tx_source_source_param_protocol;
assign ip_tx_packetizer_sink_valid = (ip_tx_source_source_valid1 & ip_tx_liteethipv4checksum_done);
assign ip_tx_source_source_ready1 = (ip_tx_packetizer_sink_ready & ip_tx_liteethipv4checksum_done);
assign ip_tx_packetizer_sink_param_target_ip = ip_tx_source_source_param_ip_address;
assign ip_tx_packetizer_sink_param_total_length = (5'd20 + ip_tx_source_source_param_length);
assign ip_tx_packetizer_sink_param_version = 3'd4;
assign ip_tx_packetizer_sink_param_ihl = 3'd5;
assign ip_tx_packetizer_sink_param_identification = 1'd0;
assign ip_tx_packetizer_sink_param_flags_offset = 1'd0;
assign ip_tx_packetizer_sink_param_ttl = ((ip_tx_source_source_param_protocol == 2'd2) ? 1'd1 : 8'd128);
assign ip_tx_packetizer_sink_param_sender_ip = 32'd3232246274;
assign ip_tx_liteethipv4checksum_header = ip_tx_packetizer_header;
assign ip_tx_packetizer_sink_param_checksum = ip_tx_liteethipv4checksum_value;
assign ip_tx_source_source_first0 = ip_tx_packetizer_source_first;
assign ip_tx_source_source_last0 = ip_tx_packetizer_source_last;
assign ip_tx_source_source_payload_data0 = ip_tx_packetizer_source_payload_data;
assign ip_tx_source_source_payload_be0 = ip_tx_packetizer_source_payload_be;
assign ip_tx_source_source_payload_error0 = ip_tx_packetizer_source_payload_error;
assign ip_tx_source_source_valid0 = (ip_tx_packetizer_source_valid & ip_tx_fsm_is_ongoing0);
assign ip_tx_packetizer_source_ready = (ip_tx_fsm_is_ongoing1 | (ip_tx_fsm_is_ongoing2 & ip_tx_source_source_ready0));
assign ip_tx_source_source_payload_ethernet_type = 12'd2048;
assign ip_tx_source_source_payload_target_mac = ip_tx_target_mac;
assign ip_tx_source_source_payload_sender_mac = 42'd2943904740353;
assign arp_table_request_request_payload_ip_address = ip_tx_source_source_param_ip_address;
assign ip_tx_pipe_valid_sink_ready = ((~ip_tx_pipe_valid_source_valid) | ip_tx_pipe_valid_source_ready);
assign ip_tx_pipe_valid_sink_valid = ip_tx_sink_sink_valid1;
assign ip_tx_sink_sink_ready1 = ip_tx_pipe_valid_sink_ready;
assign ip_tx_pipe_valid_sink_first = ip_tx_sink_sink_first1;
assign ip_tx_pipe_valid_sink_last = ip_tx_sink_sink_last1;
assign ip_tx_pipe_valid_sink_payload_data = ip_tx_sink_sink_payload_data1;
assign ip_tx_pipe_valid_sink_payload_be = ip_tx_sink_sink_payload_be1;
assign ip_tx_pipe_valid_sink_payload_error = ip_tx_sink_sink_payload_error1;
assign ip_tx_pipe_valid_sink_param_length = ip_tx_sink_sink_param_length1;
assign ip_tx_pipe_valid_sink_param_protocol = ip_tx_sink_sink_param_protocol1;
assign ip_tx_pipe_valid_sink_param_ip_address = ip_tx_sink_sink_param_ip_address1;
assign ip_tx_source_source_valid1 = ip_tx_pipe_valid_source_valid;
assign ip_tx_pipe_valid_source_ready = ip_tx_source_source_ready1;
assign ip_tx_source_source_first1 = ip_tx_pipe_valid_source_first;
assign ip_tx_source_source_last1 = ip_tx_pipe_valid_source_last;
assign ip_tx_source_source_payload_data1 = ip_tx_pipe_valid_source_payload_data;
assign ip_tx_source_source_payload_be1 = ip_tx_pipe_valid_source_payload_be;
assign ip_tx_source_source_payload_error1 = ip_tx_pipe_valid_source_payload_error;
assign ip_tx_source_source_param_length = ip_tx_pipe_valid_source_param_length;
assign ip_tx_source_source_param_protocol = ip_tx_pipe_valid_source_param_protocol;
assign ip_tx_source_source_param_ip_address = ip_tx_pipe_valid_source_param_ip_address;
assign ip_tx_liteethipv4checksum_s_next0 = (ip_tx_liteethipv4checksum_r + ip_tx_liteethipv4checksum_header[15:0]);
assign ip_tx_liteethipv4checksum_s_next1 = (ip_tx_liteethipv4checksum_r_next0 + ip_tx_liteethipv4checksum_header[31:16]);
assign ip_tx_liteethipv4checksum_s_next2 = (ip_tx_liteethipv4checksum_r_next1 + ip_tx_liteethipv4checksum_header[47:32]);
assign ip_tx_liteethipv4checksum_s_next3 = (ip_tx_liteethipv4checksum_r_next2 + ip_tx_liteethipv4checksum_header[63:48]);
assign ip_tx_liteethipv4checksum_s_next4 = (ip_tx_liteethipv4checksum_r_next3 + ip_tx_liteethipv4checksum_header[79:64]);
assign ip_tx_liteethipv4checksum_s_next5 = (ip_tx_liteethipv4checksum_r_next4 + ip_tx_liteethipv4checksum_header[111:96]);
assign ip_tx_liteethipv4checksum_s_next6 = (ip_tx_liteethipv4checksum_r_next5 + ip_tx_liteethipv4checksum_header[127:112]);
assign ip_tx_liteethipv4checksum_s_next7 = (ip_tx_liteethipv4checksum_r_next6 + ip_tx_liteethipv4checksum_header[143:128]);
assign ip_tx_liteethipv4checksum_s_next8 = (ip_tx_liteethipv4checksum_r_next7 + ip_tx_liteethipv4checksum_header[159:144]);
assign ip_tx_liteethipv4checksum_value = (~{ip_tx_liteethipv4checksum_r_next8[7:0], ip_tx_liteethipv4checksum_r_next8[15:8]});
assign ip_tx_liteethipv4checksum_counter_ce = (~ip_tx_liteethipv4checksum_done);
assign ip_tx_liteethipv4checksum_done = (ip_tx_liteethipv4checksum_counter == 4'd9);

// synthesis translate_off
reg dummy_d_37;
// synthesis translate_on
always @(*) begin
	ip_tx_packetizer_header <= 160'd0;
	ip_tx_packetizer_header[95:80] <= {ip_tx_packetizer_sink_param_checksum[7:0], ip_tx_packetizer_sink_param_checksum[15:8]};
	ip_tx_packetizer_header[63:48] <= {ip_tx_packetizer_sink_param_flags_offset[7:0], ip_tx_packetizer_sink_param_flags_offset[15:8]};
	ip_tx_packetizer_header[47:32] <= {ip_tx_packetizer_sink_param_identification[7:0], ip_tx_packetizer_sink_param_identification[15:8]};
	ip_tx_packetizer_header[3:0] <= {ip_tx_packetizer_sink_param_ihl[3:0]};
	ip_tx_packetizer_header[79:72] <= {ip_tx_packetizer_sink_param_protocol[7:0]};
	ip_tx_packetizer_header[127:96] <= {ip_tx_packetizer_sink_param_sender_ip[7:0], ip_tx_packetizer_sink_param_sender_ip[15:8], ip_tx_packetizer_sink_param_sender_ip[23:16], ip_tx_packetizer_sink_param_sender_ip[31:24]};
	ip_tx_packetizer_header[159:128] <= {ip_tx_packetizer_sink_param_target_ip[7:0], ip_tx_packetizer_sink_param_target_ip[15:8], ip_tx_packetizer_sink_param_target_ip[23:16], ip_tx_packetizer_sink_param_target_ip[31:24]};
	ip_tx_packetizer_header[31:16] <= {ip_tx_packetizer_sink_param_total_length[7:0], ip_tx_packetizer_sink_param_total_length[15:8]};
	ip_tx_packetizer_header[71:64] <= {ip_tx_packetizer_sink_param_ttl[7:0]};
	ip_tx_packetizer_header[7:4] <= {ip_tx_packetizer_sink_param_version[3:0]};
// synthesis translate_off
	dummy_d_37 <= dummy_s;
// synthesis translate_on
end
assign ip_tx_packetizer_source_payload_be = ip_tx_packetizer_source_be;

// synthesis translate_off
reg dummy_d_38;
// synthesis translate_on
always @(*) begin
	ip_tx_packetizer_source_payload_error <= 1'd0;
	if (ip_tx_packetizer_is_ongoing) begin
		ip_tx_packetizer_source_payload_error <= ip_tx_packetizer_sink_payload_error;
	end
// synthesis translate_off
	dummy_d_38 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_39;
// synthesis translate_on
always @(*) begin
	ip_tx_packetizer_sink_ready <= 1'd0;
	ip_tx_packetizer_source_valid <= 1'd0;
	ip_tx_packetizer_source_first <= 1'd0;
	ip_tx_packetizer_source_last <= 1'd0;
	ip_tx_packetizer_source_payload_data <= 8'd0;
	ip_tx_packetizer_source_be <= 1'd1;
	ip_tx_packetizer_is_ongoing <= 1'd0;
	liteethip_liteethiptx_liteethipv4packetizer_next_state <= 1'd0;
	ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value <= 5'd0;
	ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value_ce <= 1'd0;
	liteethip_liteethiptx_liteethipv4packetizer_next_state <= liteethip_liteethiptx_liteethipv4packetizer_state;
	case (liteethip_liteethiptx_liteethipv4packetizer_state)
		1'd0: begin
			ip_tx_packetizer_is_ongoing <= 1'd1;
			ip_tx_packetizer_source_valid <= ip_tx_packetizer_sink_valid;
			ip_tx_packetizer_sink_ready <= ip_tx_packetizer_source_ready;
			ip_tx_packetizer_source_last <= ip_tx_packetizer_sink_last;
			ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_sink_payload_data;
			ip_tx_packetizer_source_be <= ip_tx_packetizer_sink_payload_be;
			if (((ip_tx_packetizer_source_valid & ip_tx_packetizer_source_ready) & ip_tx_packetizer_source_last)) begin
				liteethip_liteethiptx_liteethipv4packetizer_next_state <= 1'd1;
			end
		end
		default: begin
			ip_tx_packetizer_sink_ready <= (ip_tx_packetizer_count == 1'd0);
			case (ip_tx_packetizer_count)
				1'd0: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[7:0];
				end
				1'd1: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[15:8];
				end
				2'd2: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[23:16];
				end
				2'd3: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[31:24];
				end
				3'd4: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[39:32];
				end
				3'd5: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[47:40];
				end
				3'd6: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[55:48];
				end
				3'd7: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[63:56];
				end
				4'd8: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[71:64];
				end
				4'd9: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[79:72];
				end
				4'd10: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[87:80];
				end
				4'd11: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[95:88];
				end
				4'd12: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[103:96];
				end
				4'd13: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[111:104];
				end
				4'd14: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[119:112];
				end
				4'd15: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[127:120];
				end
				5'd16: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[135:128];
				end
				5'd17: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[143:136];
				end
				5'd18: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[151:144];
				end
				5'd19: begin
					ip_tx_packetizer_source_payload_data <= ip_tx_packetizer_header[159:152];
				end
			endcase
			if ((ip_tx_packetizer_count == 5'd19)) begin
				ip_tx_packetizer_source_be <= 1'd1;
			end
			if ((ip_tx_packetizer_sink_valid & 1'd1)) begin
				ip_tx_packetizer_source_valid <= 1'd1;
				ip_tx_packetizer_source_first <= (ip_tx_packetizer_count == 1'd0);
				ip_tx_packetizer_source_last <= ((ip_tx_packetizer_count == 5'd19) & (ip_tx_packetizer_sink_last & (ip_tx_packetizer_sink_payload_be == 1'd0)));
				ip_tx_packetizer_sink_ready <= (((ip_tx_packetizer_count == 5'd19) & (ip_tx_packetizer_sink_last & (ip_tx_packetizer_sink_payload_be == 1'd0))) & ip_tx_packetizer_source_ready);
				if (ip_tx_packetizer_source_ready) begin
					if ((ip_tx_packetizer_count == 5'd19)) begin
						ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value <= 1'd0;
						ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value_ce <= 1'd1;
						if ((~ip_tx_packetizer_source_last)) begin
							liteethip_liteethiptx_liteethipv4packetizer_next_state <= 1'd0;
						end
					end else begin
						ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value <= (ip_tx_packetizer_count + 1'd1);
						ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value_ce <= 1'd1;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_39 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_40;
// synthesis translate_on
always @(*) begin
	arp_table_request_request_valid <= 1'd0;
	arp_table_response_response_ready <= 1'd0;
	ip_tx_target_unreachable <= 1'd0;
	ip_tx_fsm_is_ongoing0 <= 1'd0;
	ip_tx_fsm_is_ongoing1 <= 1'd0;
	ip_tx_fsm_is_ongoing2 <= 1'd0;
	liteethip_liteethiptx_fsm_next_state <= 3'd0;
	ip_tx_target_mac_liteethip_fsm_next_value <= 48'd0;
	ip_tx_target_mac_liteethip_fsm_next_value_ce <= 1'd0;
	liteethip_liteethiptx_fsm_next_state <= liteethip_liteethiptx_fsm_state;
	case (liteethip_liteethiptx_fsm_state)
		1'd0: begin
			ip_tx_fsm_is_ongoing0 <= 1'd1;
			ip_tx_fsm_is_ongoing2 <= 1'd1;
			if (((ip_tx_source_source_valid0 & ip_tx_source_source_last0) & ip_tx_source_source_ready0)) begin
				liteethip_liteethiptx_fsm_next_state <= 2'd2;
			end
		end
		1'd1: begin
			ip_tx_fsm_is_ongoing1 <= 1'd1;
			if (((ip_tx_packetizer_source_valid & ip_tx_packetizer_source_last) & ip_tx_packetizer_source_ready)) begin
				liteethip_liteethiptx_fsm_next_state <= 2'd2;
			end
		end
		2'd3: begin
			arp_table_request_request_valid <= 1'd1;
			if ((arp_table_request_request_valid & arp_table_request_request_ready)) begin
				liteethip_liteethiptx_fsm_next_state <= 3'd4;
			end
		end
		3'd4: begin
			if (arp_table_response_response_valid) begin
				ip_tx_target_mac_liteethip_fsm_next_value <= arp_table_response_response_payload_mac_address;
				ip_tx_target_mac_liteethip_fsm_next_value_ce <= 1'd1;
				arp_table_response_response_ready <= 1'd1;
				if (arp_table_response_response_payload_failed) begin
					ip_tx_target_unreachable <= 1'd1;
					liteethip_liteethiptx_fsm_next_state <= 1'd1;
				end else begin
					liteethip_liteethiptx_fsm_next_state <= 1'd0;
				end
			end
		end
		default: begin
			if (ip_tx_packetizer_source_valid) begin
				if ((ip_tx_source_source_param_ip_address[7:0] == 8'd255)) begin
					ip_tx_target_mac_liteethip_fsm_next_value <= 48'd281474976710655;
					ip_tx_target_mac_liteethip_fsm_next_value_ce <= 1'd1;
					liteethip_liteethiptx_fsm_next_state <= 1'd0;
				end else begin
					if ((ip_tx_source_source_param_ip_address[31:28] == 4'd14)) begin
						ip_tx_target_mac_liteethip_fsm_next_value <= {24'd65630, 1'd0, ip_tx_source_source_param_ip_address[22:0]};
						ip_tx_target_mac_liteethip_fsm_next_value_ce <= 1'd1;
						liteethip_liteethiptx_fsm_next_state <= 1'd0;
					end else begin
						liteethip_liteethiptx_fsm_next_state <= 2'd3;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_40 <= dummy_s;
// synthesis translate_on
end
assign ip_rx_depacketizer_sink_valid = ip_rx_sink_sink_valid;
assign ip_rx_sink_sink_ready = ip_rx_depacketizer_sink_ready;
assign ip_rx_depacketizer_sink_first = ip_rx_sink_sink_first;
assign ip_rx_depacketizer_sink_last = ip_rx_sink_sink_last;
assign ip_rx_depacketizer_sink_payload_ethernet_type = ip_rx_sink_sink_payload_ethernet_type;
assign ip_rx_depacketizer_sink_payload_sender_mac = ip_rx_sink_sink_payload_sender_mac;
assign ip_rx_depacketizer_sink_payload_target_mac = ip_rx_sink_sink_payload_target_mac;
assign ip_rx_depacketizer_sink_payload_data = ip_rx_sink_sink_payload_data;
assign ip_rx_depacketizer_sink_payload_be = ip_rx_sink_sink_payload_be;
assign ip_rx_depacketizer_sink_payload_error = ip_rx_sink_sink_payload_error;
assign ip_rx_liteethipv4checksum_header = ip_rx_depacketizer_header;
assign ip_rx_reset = (~ip_rx_depacketizer_source_valid);
assign ip_rx_ce = 1'd1;
assign ip_rx_accepted = (((((((ip_rx_depacketizer_source_param_target_ip == 32'd3232246274) | 1'd1) & (ip_rx_depacketizer_source_param_version == 3'd4)) & (ip_rx_depacketizer_source_param_ihl == 3'd5)) & (ip_rx_depacketizer_source_param_total_length > 5'd20)) & ((ip_rx_depacketizer_source_param_flags_offset & 14'd16383) == 1'd0)) & (ip_rx_liteethipv4checksum_value == 1'd0));
assign ip_rx_receiving = ip_rx_fsm_is_ongoing0;
assign ip_rx_source_source_valid = (ip_rx_depacketizer_source_valid & ip_rx_receiving);
assign ip_rx_depacketizer_source_ready = (ip_rx_fsm_is_ongoing1 | (ip_rx_receiving & ip_rx_source_source_ready));
assign ip_rx_source_source_last = ip_rx_depacketizer_source_last;
assign ip_rx_source_source_payload_data = ip_rx_depacketizer_source_payload_data;
assign ip_rx_source_source_payload_be = ip_rx_depacketizer_source_payload_be;
assign ip_rx_source_source_payload_error = ip_rx_depacketizer_source_payload_error;
assign ip_rx_source_source_param_protocol = ip_rx_depacketizer_source_param_protocol;
assign ip_rx_source_source_param_length = (ip_rx_depacketizer_source_param_total_length - 5'd20);
assign ip_rx_source_source_param_ip_address = ip_rx_depacketizer_source_param_sender_ip;
assign ip_rx_depacketizer_header = ip_rx_depacketizer_sr;
assign ip_rx_depacketizer_source_param_checksum = {rhs_slice_proxy43[7:0], rhs_slice_proxy42[15:8]};
assign ip_rx_depacketizer_source_param_flags_offset = {rhs_slice_proxy45[7:0], rhs_slice_proxy44[15:8]};
assign ip_rx_depacketizer_source_param_identification = {rhs_slice_proxy47[7:0], rhs_slice_proxy46[15:8]};
assign ip_rx_depacketizer_source_param_ihl = {rhs_slice_proxy48[3:0]};
assign ip_rx_depacketizer_source_param_protocol = {rhs_slice_proxy49[7:0]};
assign ip_rx_depacketizer_source_param_sender_ip = {rhs_slice_proxy53[7:0], rhs_slice_proxy52[15:8], rhs_slice_proxy51[23:16], rhs_slice_proxy50[31:24]};
assign ip_rx_depacketizer_source_param_target_ip = {rhs_slice_proxy57[7:0], rhs_slice_proxy56[15:8], rhs_slice_proxy55[23:16], rhs_slice_proxy54[31:24]};
assign ip_rx_depacketizer_source_param_total_length = {rhs_slice_proxy59[7:0], rhs_slice_proxy58[15:8]};
assign ip_rx_depacketizer_source_param_ttl = {rhs_slice_proxy60[7:0]};
assign ip_rx_depacketizer_source_param_version = {rhs_slice_proxy61[3:0]};
assign ip_rx_depacketizer_source_payload_error = ip_rx_depacketizer_sink_payload_error;
assign ip_rx_depacketizer_sr_shift = ((((ip_rx_depacketizer_is_ongoing0 | ip_rx_depacketizer_is_ongoing1) & ip_rx_depacketizer_sink_valid) & ip_rx_depacketizer_sink_ready) & (ip_rx_depacketizer_skip_cnt >= 1'd0));

// synthesis translate_off
reg dummy_d_41;
// synthesis translate_on
always @(*) begin
	ip_rx_depacketizer_sink_ready <= 1'd0;
	ip_rx_depacketizer_source_valid <= 1'd0;
	ip_rx_depacketizer_source_first <= 1'd0;
	ip_rx_depacketizer_source_last <= 1'd0;
	ip_rx_depacketizer_source_payload_data <= 8'd0;
	ip_rx_depacketizer_source_payload_be <= 1'd0;
	ip_rx_depacketizer_is_ongoing0 <= 1'd0;
	ip_rx_depacketizer_is_ongoing1 <= 1'd0;
	liteethip_liteethiprx_liteethipv4depacketizer_next_state <= 2'd0;
	ip_rx_depacketizer_count_liteethip_next_value <= 5'd0;
	ip_rx_depacketizer_count_liteethip_next_value_ce <= 1'd0;
	ip_rx_depacketizer_data_copy_first_liteethip_t_next_value0 <= 1'd0;
	ip_rx_depacketizer_data_copy_first_liteethip_t_next_value_ce0 <= 1'd0;
	ip_rx_depacketizer_send_dummy_liteethip_t_next_value1 <= 1'd0;
	ip_rx_depacketizer_send_dummy_liteethip_t_next_value_ce1 <= 1'd0;
	ip_rx_depacketizer_skip_cnt_liteethip_f_next_value <= 1'd0;
	ip_rx_depacketizer_skip_cnt_liteethip_f_next_value_ce <= 1'd0;
	ip_rx_depacketizer_source_payload_data <= ip_rx_depacketizer_sink_payload_data;
	ip_rx_depacketizer_source_payload_be <= ip_rx_depacketizer_sink_payload_be;
	liteethip_liteethiprx_liteethipv4depacketizer_next_state <= liteethip_liteethiprx_liteethipv4depacketizer_state;
	case (liteethip_liteethiprx_liteethipv4depacketizer_state)
		1'd1: begin
			ip_rx_depacketizer_is_ongoing1 <= 1'd1;
			ip_rx_depacketizer_sink_ready <= 1'd1;
			if ((ip_rx_depacketizer_sink_valid & ip_rx_depacketizer_sink_ready)) begin
				ip_rx_depacketizer_count_liteethip_next_value <= (ip_rx_depacketizer_count + 1'd1);
				ip_rx_depacketizer_count_liteethip_next_value_ce <= 1'd1;
				if ((ip_rx_depacketizer_count == 5'd19)) begin
					ip_rx_depacketizer_data_copy_first_liteethip_t_next_value0 <= 1'd1;
					ip_rx_depacketizer_data_copy_first_liteethip_t_next_value_ce0 <= 1'd1;
					liteethip_liteethiprx_liteethipv4depacketizer_next_state <= 2'd2;
					ip_rx_depacketizer_count_liteethip_next_value <= (ip_rx_depacketizer_count + 1'd1);
					ip_rx_depacketizer_count_liteethip_next_value_ce <= 1'd1;
				end
				if ((ip_rx_depacketizer_sink_last & ip_rx_depacketizer_sink_ready)) begin
					liteethip_liteethiprx_liteethipv4depacketizer_next_state <= 1'd0;
				end
			end
		end
		2'd2: begin
			if ((1'd0 & ip_rx_depacketizer_send_dummy)) begin
				ip_rx_depacketizer_source_valid <= 1'd1;
				ip_rx_depacketizer_source_payload_data <= 1'd0;
				ip_rx_depacketizer_source_first <= 1'd1;
				ip_rx_depacketizer_source_last <= 1'd0;
				ip_rx_depacketizer_source_payload_be <= 1'd1;
				ip_rx_depacketizer_sink_ready <= 1'd0;
				if (ip_rx_depacketizer_source_ready) begin
					ip_rx_depacketizer_send_dummy_liteethip_t_next_value1 <= 1'd0;
					ip_rx_depacketizer_send_dummy_liteethip_t_next_value_ce1 <= 1'd1;
				end
			end else begin
				ip_rx_depacketizer_source_valid <= ip_rx_depacketizer_sink_valid;
				ip_rx_depacketizer_source_last <= ip_rx_depacketizer_sink_last;
				ip_rx_depacketizer_source_first <= (ip_rx_depacketizer_data_copy_first & ip_rx_depacketizer_sink_valid);
				ip_rx_depacketizer_sink_ready <= ip_rx_depacketizer_source_ready;
				if ((ip_rx_depacketizer_source_valid & ip_rx_depacketizer_source_ready)) begin
					ip_rx_depacketizer_data_copy_first_liteethip_t_next_value0 <= 1'd0;
					ip_rx_depacketizer_data_copy_first_liteethip_t_next_value_ce0 <= 1'd1;
					if (ip_rx_depacketizer_source_last) begin
						ip_rx_depacketizer_skip_cnt_liteethip_f_next_value <= 1'd0;
						ip_rx_depacketizer_skip_cnt_liteethip_f_next_value_ce <= 1'd1;
						if (1'd0) begin
							ip_rx_depacketizer_send_dummy_liteethip_t_next_value1 <= 1'd1;
							ip_rx_depacketizer_send_dummy_liteethip_t_next_value_ce1 <= 1'd1;
						end
						liteethip_liteethiprx_liteethipv4depacketizer_next_state <= 1'd0;
					end
				end
			end
		end
		default: begin
			ip_rx_depacketizer_is_ongoing0 <= 1'd1;
			ip_rx_depacketizer_sink_ready <= 1'd1;
			ip_rx_depacketizer_count_liteethip_next_value <= 1'd1;
			ip_rx_depacketizer_count_liteethip_next_value_ce <= 1'd1;
			if ((ip_rx_depacketizer_sink_valid & ip_rx_depacketizer_sink_ready)) begin
				liteethip_liteethiprx_liteethipv4depacketizer_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_41 <= dummy_s;
// synthesis translate_on
end
assign ip_rx_liteethipv4checksum_s_next0 = (ip_rx_liteethipv4checksum_r + ip_rx_liteethipv4checksum_header[15:0]);
assign ip_rx_liteethipv4checksum_s_next1 = (ip_rx_liteethipv4checksum_r_next0 + ip_rx_liteethipv4checksum_header[31:16]);
assign ip_rx_liteethipv4checksum_s_next2 = (ip_rx_liteethipv4checksum_r_next1 + ip_rx_liteethipv4checksum_header[47:32]);
assign ip_rx_liteethipv4checksum_s_next3 = (ip_rx_liteethipv4checksum_r_next2 + ip_rx_liteethipv4checksum_header[63:48]);
assign ip_rx_liteethipv4checksum_s_next4 = (ip_rx_liteethipv4checksum_r_next3 + ip_rx_liteethipv4checksum_header[79:64]);
assign ip_rx_liteethipv4checksum_s_next5 = (ip_rx_liteethipv4checksum_r_next4 + ip_rx_liteethipv4checksum_header[95:80]);
assign ip_rx_liteethipv4checksum_s_next6 = (ip_rx_liteethipv4checksum_r_next5 + ip_rx_liteethipv4checksum_header[111:96]);
assign ip_rx_liteethipv4checksum_s_next7 = (ip_rx_liteethipv4checksum_r_next6 + ip_rx_liteethipv4checksum_header[127:112]);
assign ip_rx_liteethipv4checksum_s_next8 = (ip_rx_liteethipv4checksum_r_next7 + ip_rx_liteethipv4checksum_header[143:128]);
assign ip_rx_liteethipv4checksum_s_next9 = (ip_rx_liteethipv4checksum_r_next8 + ip_rx_liteethipv4checksum_header[159:144]);
assign ip_rx_liteethipv4checksum_value = (~{ip_rx_liteethipv4checksum_r_next9[7:0], ip_rx_liteethipv4checksum_r_next9[15:8]});
assign ip_rx_liteethipv4checksum_counter_ce = (~ip_rx_liteethipv4checksum_done);
assign ip_rx_liteethipv4checksum_done = (ip_rx_liteethipv4checksum_counter == 4'd11);

// synthesis translate_off
reg dummy_d_42;
// synthesis translate_on
always @(*) begin
	ip_rx_fsm_is_ongoing0 <= 1'd0;
	ip_rx_fsm_is_ongoing1 <= 1'd0;
	liteethip_liteethiprx_fsm_next_state <= 2'd0;
	liteethip_liteethiprx_fsm_next_state <= liteethip_liteethiprx_fsm_state;
	case (liteethip_liteethiprx_fsm_state)
		1'd0: begin
			ip_rx_fsm_is_ongoing0 <= 1'd1;
			if ((ip_rx_source_source_valid & ip_rx_source_source_ready)) begin
				if (ip_rx_source_source_last) begin
					liteethip_liteethiprx_fsm_next_state <= 1'd1;
				end
			end
		end
		2'd2: begin
			ip_rx_fsm_is_ongoing1 <= 1'd1;
			if (((ip_rx_depacketizer_source_valid & ip_rx_depacketizer_source_last) & ip_rx_depacketizer_source_ready)) begin
				liteethip_liteethiprx_fsm_next_state <= 1'd1;
			end
		end
		default: begin
			if ((ip_rx_depacketizer_source_valid & ip_rx_liteethipv4checksum_done)) begin
				liteethip_liteethiprx_fsm_next_state <= 2'd2;
				if (ip_rx_accepted) begin
					liteethip_liteethiprx_fsm_next_state <= 1'd0;
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_42 <= dummy_s;
// synthesis translate_on
end
assign ip_rx_buffer_pipe_valid_sink_ready = ((~ip_rx_buffer_pipe_valid_source_valid) | ip_rx_buffer_pipe_valid_source_ready);
assign ip_rx_buffer_pipe_ready_sink_ready = (~ip_rx_buffer_pipe_ready_valid);

// synthesis translate_off
reg dummy_d_43;
// synthesis translate_on
always @(*) begin
	ip_rx_buffer_pipe_ready_source_valid <= 1'd0;
	ip_rx_buffer_pipe_ready_source_first <= 1'd0;
	ip_rx_buffer_pipe_ready_source_last <= 1'd0;
	ip_rx_buffer_pipe_ready_source_payload_data <= 8'd0;
	ip_rx_buffer_pipe_ready_source_payload_be <= 1'd0;
	ip_rx_buffer_pipe_ready_source_payload_error <= 1'd0;
	ip_rx_buffer_pipe_ready_source_param_length <= 16'd0;
	ip_rx_buffer_pipe_ready_source_param_protocol <= 8'd0;
	ip_rx_buffer_pipe_ready_source_param_ip_address <= 32'd0;
	if (ip_rx_buffer_pipe_ready_valid) begin
		ip_rx_buffer_pipe_ready_source_valid <= ip_rx_buffer_pipe_ready_sink_d_valid;
		ip_rx_buffer_pipe_ready_source_first <= ip_rx_buffer_pipe_ready_sink_d_first;
		ip_rx_buffer_pipe_ready_source_last <= ip_rx_buffer_pipe_ready_sink_d_last;
		ip_rx_buffer_pipe_ready_source_payload_data <= ip_rx_buffer_pipe_ready_sink_d_payload_data;
		ip_rx_buffer_pipe_ready_source_payload_be <= ip_rx_buffer_pipe_ready_sink_d_payload_be;
		ip_rx_buffer_pipe_ready_source_payload_error <= ip_rx_buffer_pipe_ready_sink_d_payload_error;
		ip_rx_buffer_pipe_ready_source_param_length <= ip_rx_buffer_pipe_ready_sink_d_param_length;
		ip_rx_buffer_pipe_ready_source_param_protocol <= ip_rx_buffer_pipe_ready_sink_d_param_protocol;
		ip_rx_buffer_pipe_ready_source_param_ip_address <= ip_rx_buffer_pipe_ready_sink_d_param_ip_address;
	end else begin
		ip_rx_buffer_pipe_ready_source_valid <= ip_rx_buffer_pipe_ready_sink_valid;
		ip_rx_buffer_pipe_ready_source_first <= ip_rx_buffer_pipe_ready_sink_first;
		ip_rx_buffer_pipe_ready_source_last <= ip_rx_buffer_pipe_ready_sink_last;
		ip_rx_buffer_pipe_ready_source_payload_data <= ip_rx_buffer_pipe_ready_sink_payload_data;
		ip_rx_buffer_pipe_ready_source_payload_be <= ip_rx_buffer_pipe_ready_sink_payload_be;
		ip_rx_buffer_pipe_ready_source_payload_error <= ip_rx_buffer_pipe_ready_sink_payload_error;
		ip_rx_buffer_pipe_ready_source_param_length <= ip_rx_buffer_pipe_ready_sink_param_length;
		ip_rx_buffer_pipe_ready_source_param_protocol <= ip_rx_buffer_pipe_ready_sink_param_protocol;
		ip_rx_buffer_pipe_ready_source_param_ip_address <= ip_rx_buffer_pipe_ready_sink_param_ip_address;
	end
// synthesis translate_off
	dummy_d_43 <= dummy_s;
// synthesis translate_on
end
assign ip_rx_buffer_pipe_valid_sink_valid = ip_rx_buffer_sink_sink_valid;
assign ip_rx_buffer_sink_sink_ready = ip_rx_buffer_pipe_valid_sink_ready;
assign ip_rx_buffer_pipe_valid_sink_first = ip_rx_buffer_sink_sink_first;
assign ip_rx_buffer_pipe_valid_sink_last = ip_rx_buffer_sink_sink_last;
assign ip_rx_buffer_pipe_valid_sink_payload_data = ip_rx_buffer_sink_sink_payload_data;
assign ip_rx_buffer_pipe_valid_sink_payload_be = ip_rx_buffer_sink_sink_payload_be;
assign ip_rx_buffer_pipe_valid_sink_payload_error = ip_rx_buffer_sink_sink_payload_error;
assign ip_rx_buffer_pipe_valid_sink_param_length = ip_rx_buffer_sink_sink_param_length;
assign ip_rx_buffer_pipe_valid_sink_param_protocol = ip_rx_buffer_sink_sink_param_protocol;
assign ip_rx_buffer_pipe_valid_sink_param_ip_address = ip_rx_buffer_sink_sink_param_ip_address;
assign ip_rx_buffer_pipe_ready_sink_valid = ip_rx_buffer_pipe_valid_source_valid;
assign ip_rx_buffer_pipe_valid_source_ready = ip_rx_buffer_pipe_ready_sink_ready;
assign ip_rx_buffer_pipe_ready_sink_first = ip_rx_buffer_pipe_valid_source_first;
assign ip_rx_buffer_pipe_ready_sink_last = ip_rx_buffer_pipe_valid_source_last;
assign ip_rx_buffer_pipe_ready_sink_payload_data = ip_rx_buffer_pipe_valid_source_payload_data;
assign ip_rx_buffer_pipe_ready_sink_payload_be = ip_rx_buffer_pipe_valid_source_payload_be;
assign ip_rx_buffer_pipe_ready_sink_payload_error = ip_rx_buffer_pipe_valid_source_payload_error;
assign ip_rx_buffer_pipe_ready_sink_param_length = ip_rx_buffer_pipe_valid_source_param_length;
assign ip_rx_buffer_pipe_ready_sink_param_protocol = ip_rx_buffer_pipe_valid_source_param_protocol;
assign ip_rx_buffer_pipe_ready_sink_param_ip_address = ip_rx_buffer_pipe_valid_source_param_ip_address;
assign ip_rx_buffer_source_source_valid = ip_rx_buffer_pipe_ready_source_valid;
assign ip_rx_buffer_pipe_ready_source_ready = ip_rx_buffer_source_source_ready;
assign ip_rx_buffer_source_source_first = ip_rx_buffer_pipe_ready_source_first;
assign ip_rx_buffer_source_source_last = ip_rx_buffer_pipe_ready_source_last;
assign ip_rx_buffer_source_source_payload_data = ip_rx_buffer_pipe_ready_source_payload_data;
assign ip_rx_buffer_source_source_payload_be = ip_rx_buffer_pipe_ready_source_payload_be;
assign ip_rx_buffer_source_source_payload_error = ip_rx_buffer_pipe_ready_source_payload_error;
assign ip_rx_buffer_source_source_param_length = ip_rx_buffer_pipe_ready_source_param_length;
assign ip_rx_buffer_source_source_param_protocol = ip_rx_buffer_pipe_ready_source_param_protocol;
assign ip_rx_buffer_source_source_param_ip_address = ip_rx_buffer_pipe_ready_source_param_ip_address;

// synthesis translate_off
reg dummy_d_44;
// synthesis translate_on
always @(*) begin
	liteethip_sel0 <= 2'd0;
	if ((ip_crossbar_sink_param_protocol == 1'd1)) begin
		liteethip_sel0 <= 1'd1;
	end
	if ((ip_crossbar_sink_param_protocol == 5'd17)) begin
		liteethip_sel0 <= 2'd2;
	end
// synthesis translate_off
	dummy_d_44 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_45;
// synthesis translate_on
always @(*) begin
	liteethip_request <= 2'd0;
	liteethip_request[0] <= liteethip_status0_ongoing0;
	liteethip_request[1] <= liteethip_status1_ongoing0;
// synthesis translate_off
	dummy_d_45 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_46;
// synthesis translate_on
always @(*) begin
	ip_crossbar_source_valid <= 1'd0;
	ip_crossbar_source_first <= 1'd0;
	ip_crossbar_source_last <= 1'd0;
	ip_crossbar_source_payload_data <= 8'd0;
	ip_crossbar_source_payload_be <= 1'd0;
	ip_crossbar_source_payload_error <= 1'd0;
	ip_crossbar_source_param_length <= 16'd0;
	ip_crossbar_source_param_protocol <= 8'd0;
	ip_crossbar_source_param_ip_address <= 32'd0;
	icmp_sink_ready <= 1'd0;
	ip_port_sink_ready <= 1'd0;
	case (liteethip_grant)
		1'd0: begin
			ip_crossbar_source_valid <= icmp_sink_valid;
			icmp_sink_ready <= ip_crossbar_source_ready;
			ip_crossbar_source_first <= icmp_sink_first;
			ip_crossbar_source_last <= icmp_sink_last;
			ip_crossbar_source_payload_data <= icmp_sink_payload_data;
			ip_crossbar_source_payload_be <= icmp_sink_payload_be;
			ip_crossbar_source_payload_error <= icmp_sink_payload_error;
			ip_crossbar_source_param_length <= icmp_sink_param_length;
			ip_crossbar_source_param_protocol <= icmp_sink_param_protocol;
			ip_crossbar_source_param_ip_address <= icmp_sink_param_ip_address;
		end
		1'd1: begin
			ip_crossbar_source_valid <= ip_port_sink_valid;
			ip_port_sink_ready <= ip_crossbar_source_ready;
			ip_crossbar_source_first <= ip_port_sink_first;
			ip_crossbar_source_last <= ip_port_sink_last;
			ip_crossbar_source_payload_data <= ip_port_sink_payload_data;
			ip_crossbar_source_payload_be <= ip_port_sink_payload_be;
			ip_crossbar_source_payload_error <= ip_port_sink_payload_error;
			ip_crossbar_source_param_length <= ip_port_sink_param_length;
			ip_crossbar_source_param_protocol <= ip_port_sink_param_protocol;
			ip_crossbar_source_param_ip_address <= ip_port_sink_param_ip_address;
		end
	endcase
// synthesis translate_off
	dummy_d_46 <= dummy_s;
// synthesis translate_on
end
assign liteethip_status0_last = ((icmp_sink_valid & icmp_sink_last) & icmp_sink_ready);
assign liteethip_status0_ongoing0 = ((icmp_sink_valid | liteethip_status0_ongoing1) & (~liteethip_status0_last));
assign liteethip_status1_last = ((ip_port_sink_valid & ip_port_sink_last) & ip_port_sink_ready);
assign liteethip_status1_ongoing0 = ((ip_port_sink_valid | liteethip_status1_ongoing1) & (~liteethip_status1_last));

// synthesis translate_off
reg dummy_d_47;
// synthesis translate_on
always @(*) begin
	liteethip_sel1 <= 2'd0;
	if ((~liteethip_sel_locked)) begin
		liteethip_sel1 <= liteethip_sel0;
	end else begin
		liteethip_sel1 <= liteethip_sel_ongoing;
	end
// synthesis translate_off
	dummy_d_47 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_48;
// synthesis translate_on
always @(*) begin
	ip_crossbar_sink_ready <= 1'd0;
	icmp_source_valid <= 1'd0;
	icmp_source_first <= 1'd0;
	icmp_source_last <= 1'd0;
	icmp_source_payload_data <= 8'd0;
	icmp_source_payload_be <= 1'd0;
	icmp_source_payload_error <= 1'd0;
	icmp_source_param_length <= 16'd0;
	icmp_source_param_protocol <= 8'd0;
	icmp_source_param_ip_address <= 32'd0;
	ip_port_source_valid <= 1'd0;
	ip_port_source_first <= 1'd0;
	ip_port_source_last <= 1'd0;
	ip_port_source_payload_data <= 8'd0;
	ip_port_source_payload_be <= 1'd0;
	ip_port_source_payload_error <= 1'd0;
	ip_port_source_param_length <= 16'd0;
	ip_port_source_param_protocol <= 8'd0;
	ip_port_source_param_ip_address <= 32'd0;
	case (liteethip_sel1)
		1'd1: begin
			icmp_source_valid <= ip_crossbar_sink_valid;
			ip_crossbar_sink_ready <= icmp_source_ready;
			icmp_source_first <= ip_crossbar_sink_first;
			icmp_source_last <= ip_crossbar_sink_last;
			icmp_source_payload_data <= ip_crossbar_sink_payload_data;
			icmp_source_payload_be <= ip_crossbar_sink_payload_be;
			icmp_source_payload_error <= ip_crossbar_sink_payload_error;
			icmp_source_param_length <= ip_crossbar_sink_param_length;
			icmp_source_param_protocol <= ip_crossbar_sink_param_protocol;
			icmp_source_param_ip_address <= ip_crossbar_sink_param_ip_address;
		end
		2'd2: begin
			ip_port_source_valid <= ip_crossbar_sink_valid;
			ip_crossbar_sink_ready <= ip_port_source_ready;
			ip_port_source_first <= ip_crossbar_sink_first;
			ip_port_source_last <= ip_crossbar_sink_last;
			ip_port_source_payload_data <= ip_crossbar_sink_payload_data;
			ip_port_source_payload_be <= ip_crossbar_sink_payload_be;
			ip_port_source_payload_error <= ip_crossbar_sink_payload_error;
			ip_port_source_param_length <= ip_crossbar_sink_param_length;
			ip_port_source_param_protocol <= ip_crossbar_sink_param_protocol;
			ip_port_source_param_ip_address <= ip_crossbar_sink_param_ip_address;
		end
		default: begin
			ip_crossbar_sink_ready <= 1'd1;
		end
	endcase
// synthesis translate_off
	dummy_d_48 <= dummy_s;
// synthesis translate_on
end
assign liteethip_last = ((ip_crossbar_sink_valid & ip_crossbar_sink_last) & ip_crossbar_sink_ready);
assign liteethip_ongoing0 = ((ip_crossbar_sink_valid | liteethip_ongoing1) & (~liteethip_last));
assign icmp_echo_sink_sink_valid = icmp_rx_source_source_valid;
assign icmp_rx_source_source_ready = icmp_echo_sink_sink_ready;
assign icmp_echo_sink_sink_first = icmp_rx_source_source_first;
assign icmp_echo_sink_sink_last = icmp_rx_source_source_last;
assign icmp_echo_sink_sink_payload_data = icmp_rx_source_source_payload_data;
assign icmp_echo_sink_sink_payload_be = icmp_rx_source_source_payload_be;
assign icmp_echo_sink_sink_payload_error = icmp_rx_source_source_payload_error;
assign icmp_echo_sink_sink_param_checksum = icmp_rx_source_source_param_checksum;
assign icmp_echo_sink_sink_param_code = icmp_rx_source_source_param_code;
assign icmp_echo_sink_sink_param_msgtype = icmp_rx_source_source_param_msgtype;
assign icmp_echo_sink_sink_param_quench = icmp_rx_source_source_param_quench;
assign icmp_echo_sink_sink_param_ip_address = icmp_rx_source_source_param_ip_address;
assign icmp_echo_sink_sink_param_length = icmp_rx_source_source_param_length;
assign icmp_tx_sink_sink_valid = icmp_echo_source_source_valid;
assign icmp_echo_source_source_ready = icmp_tx_sink_sink_ready;
assign icmp_tx_sink_sink_first = icmp_echo_source_source_first;
assign icmp_tx_sink_sink_last = icmp_echo_source_source_last;
assign icmp_tx_sink_sink_payload_data = icmp_echo_source_source_payload_data;
assign icmp_tx_sink_sink_payload_be = icmp_echo_source_source_payload_be;
assign icmp_tx_sink_sink_payload_error = icmp_echo_source_source_payload_error;
assign icmp_tx_sink_sink_param_checksum = icmp_echo_source_source_param_checksum;
assign icmp_tx_sink_sink_param_code = icmp_echo_source_source_param_code;
assign icmp_tx_sink_sink_param_msgtype = icmp_echo_source_source_param_msgtype;
assign icmp_tx_sink_sink_param_quench = icmp_echo_source_source_param_quench;
assign icmp_tx_sink_sink_param_ip_address = icmp_echo_source_source_param_ip_address;
assign icmp_tx_sink_sink_param_length = icmp_echo_source_source_param_length;
assign icmp_sink_valid = icmp_tx_source_source_valid;
assign icmp_tx_source_source_ready = icmp_sink_ready;
assign icmp_sink_first = icmp_tx_source_source_first;
assign icmp_sink_last = icmp_tx_source_source_last;
assign icmp_sink_payload_data = icmp_tx_source_source_payload_data;
assign icmp_sink_payload_be = icmp_tx_source_source_payload_be;
assign icmp_sink_payload_error = icmp_tx_source_source_payload_error;
assign icmp_sink_param_length = icmp_tx_source_source_param_length;
assign icmp_sink_param_protocol = icmp_tx_source_source_param_protocol;
assign icmp_sink_param_ip_address = icmp_tx_source_source_param_ip_address;
assign icmp_rx_sink_sink_valid = icmp_source_valid;
assign icmp_source_ready = icmp_rx_sink_sink_ready;
assign icmp_rx_sink_sink_first = icmp_source_first;
assign icmp_rx_sink_sink_last = icmp_source_last;
assign icmp_rx_sink_sink_payload_data = icmp_source_payload_data;
assign icmp_rx_sink_sink_payload_be = icmp_source_payload_be;
assign icmp_rx_sink_sink_payload_error = icmp_source_payload_error;
assign icmp_rx_sink_sink_param_length = icmp_source_param_length;
assign icmp_rx_sink_sink_param_protocol = icmp_source_param_protocol;
assign icmp_rx_sink_sink_param_ip_address = icmp_source_param_ip_address;
assign icmp_tx_packetizer_sink_valid = icmp_tx_sink_sink_valid;
assign icmp_tx_sink_sink_ready = icmp_tx_packetizer_sink_ready;
assign icmp_tx_packetizer_sink_last = icmp_tx_sink_sink_last;
assign icmp_tx_packetizer_sink_payload_data = icmp_tx_sink_sink_payload_data;
assign icmp_tx_packetizer_sink_payload_be = icmp_tx_sink_sink_payload_be;
assign icmp_tx_packetizer_sink_param_checksum = icmp_tx_sink_sink_param_checksum;
assign icmp_tx_packetizer_sink_param_code = icmp_tx_sink_sink_param_code;
assign icmp_tx_packetizer_sink_param_msgtype = icmp_tx_sink_sink_param_msgtype;
assign icmp_tx_packetizer_sink_param_quench = icmp_tx_sink_sink_param_quench;
assign icmp_tx_source_source_valid = icmp_tx_packetizer_source_valid;
assign icmp_tx_packetizer_source_ready = icmp_tx_source_source_ready;
assign icmp_tx_source_source_first = icmp_tx_packetizer_source_first;
assign icmp_tx_source_source_last = icmp_tx_packetizer_source_last;
assign icmp_tx_source_source_payload_data = icmp_tx_packetizer_source_payload_data;
assign icmp_tx_source_source_payload_be = icmp_tx_packetizer_source_payload_be;
assign icmp_tx_source_source_payload_error = icmp_tx_packetizer_source_payload_error;
assign icmp_tx_source_source_param_length = (icmp_tx_active ? icmp_tx_length : (icmp_tx_sink_sink_param_length + 4'd8));
assign icmp_tx_source_source_param_protocol = 1'd1;
assign icmp_tx_source_source_param_ip_address = (icmp_tx_active ? icmp_tx_ip_address : icmp_tx_sink_sink_param_ip_address);

// synthesis translate_off
reg dummy_d_49;
// synthesis translate_on
always @(*) begin
	icmp_tx_packetizer_header <= 64'd0;
	icmp_tx_packetizer_header[31:16] <= {icmp_tx_packetizer_sink_param_checksum[7:0], icmp_tx_packetizer_sink_param_checksum[15:8]};
	icmp_tx_packetizer_header[15:8] <= {icmp_tx_packetizer_sink_param_code[7:0]};
	icmp_tx_packetizer_header[7:0] <= {icmp_tx_packetizer_sink_param_msgtype[7:0]};
	icmp_tx_packetizer_header[63:32] <= {icmp_tx_packetizer_sink_param_quench[7:0], icmp_tx_packetizer_sink_param_quench[15:8], icmp_tx_packetizer_sink_param_quench[23:16], icmp_tx_packetizer_sink_param_quench[31:24]};
// synthesis translate_off
	dummy_d_49 <= dummy_s;
// synthesis translate_on
end
assign icmp_tx_packetizer_source_payload_be = icmp_tx_packetizer_source_be;

// synthesis translate_off
reg dummy_d_50;
// synthesis translate_on
always @(*) begin
	icmp_tx_packetizer_source_payload_error <= 1'd0;
	if (icmp_tx_packetizer_is_ongoing) begin
		icmp_tx_packetizer_source_payload_error <= icmp_tx_packetizer_sink_payload_error;
	end
// synthesis translate_off
	dummy_d_50 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_51;
// synthesis translate_on
always @(*) begin
	icmp_tx_packetizer_sink_ready <= 1'd0;
	icmp_tx_packetizer_source_valid <= 1'd0;
	icmp_tx_packetizer_source_first <= 1'd0;
	icmp_tx_packetizer_source_last <= 1'd0;
	icmp_tx_packetizer_source_payload_data <= 8'd0;
	icmp_tx_packetizer_source_be <= 1'd1;
	icmp_tx_packetizer_is_ongoing <= 1'd0;
	liteethicmp_next_state <= 1'd0;
	icmp_tx_packetizer_count_liteethicmp_next_value <= 3'd0;
	icmp_tx_packetizer_count_liteethicmp_next_value_ce <= 1'd0;
	liteethicmp_next_state <= liteethicmp_state;
	case (liteethicmp_state)
		1'd0: begin
			icmp_tx_packetizer_is_ongoing <= 1'd1;
			icmp_tx_packetizer_source_valid <= icmp_tx_packetizer_sink_valid;
			icmp_tx_packetizer_sink_ready <= icmp_tx_packetizer_source_ready;
			icmp_tx_packetizer_source_last <= icmp_tx_packetizer_sink_last;
			icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_sink_payload_data;
			icmp_tx_packetizer_source_be <= icmp_tx_packetizer_sink_payload_be;
			if (((icmp_tx_packetizer_source_valid & icmp_tx_packetizer_source_ready) & icmp_tx_packetizer_source_last)) begin
				liteethicmp_next_state <= 1'd1;
			end
		end
		default: begin
			icmp_tx_packetizer_sink_ready <= (icmp_tx_packetizer_count == 1'd0);
			case (icmp_tx_packetizer_count)
				1'd0: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[7:0];
				end
				1'd1: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[15:8];
				end
				2'd2: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[23:16];
				end
				2'd3: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[31:24];
				end
				3'd4: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[39:32];
				end
				3'd5: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[47:40];
				end
				3'd6: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[55:48];
				end
				3'd7: begin
					icmp_tx_packetizer_source_payload_data <= icmp_tx_packetizer_header[63:56];
				end
			endcase
			if ((icmp_tx_packetizer_count == 3'd7)) begin
				icmp_tx_packetizer_source_be <= 1'd1;
			end
			if ((icmp_tx_packetizer_sink_valid & 1'd1)) begin
				icmp_tx_packetizer_source_valid <= 1'd1;
				icmp_tx_packetizer_source_first <= (icmp_tx_packetizer_count == 1'd0);
				icmp_tx_packetizer_source_last <= ((icmp_tx_packetizer_count == 3'd7) & (icmp_tx_packetizer_sink_last & (icmp_tx_packetizer_sink_payload_be == 1'd0)));
				icmp_tx_packetizer_sink_ready <= (((icmp_tx_packetizer_count == 3'd7) & (icmp_tx_packetizer_sink_last & (icmp_tx_packetizer_sink_payload_be == 1'd0))) & icmp_tx_packetizer_source_ready);
				if (icmp_tx_packetizer_source_ready) begin
					if ((icmp_tx_packetizer_count == 3'd7)) begin
						icmp_tx_packetizer_count_liteethicmp_next_value <= 1'd0;
						icmp_tx_packetizer_count_liteethicmp_next_value_ce <= 1'd1;
						if ((~icmp_tx_packetizer_source_last)) begin
							liteethicmp_next_state <= 1'd0;
						end
					end else begin
						icmp_tx_packetizer_count_liteethicmp_next_value <= (icmp_tx_packetizer_count + 1'd1);
						icmp_tx_packetizer_count_liteethicmp_next_value_ce <= 1'd1;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_51 <= dummy_s;
// synthesis translate_on
end
assign icmp_rx_depacketizer_sink_valid = icmp_rx_sink_sink_valid;
assign icmp_rx_sink_sink_ready = icmp_rx_depacketizer_sink_ready;
assign icmp_rx_depacketizer_sink_first = icmp_rx_sink_sink_first;
assign icmp_rx_depacketizer_sink_last = icmp_rx_sink_sink_last;
assign icmp_rx_depacketizer_sink_payload_data = icmp_rx_sink_sink_payload_data;
assign icmp_rx_depacketizer_sink_payload_be = icmp_rx_sink_sink_payload_be;
assign icmp_rx_depacketizer_sink_payload_error = icmp_rx_sink_sink_payload_error;
assign icmp_rx_depacketizer_sink_param_length = icmp_rx_sink_sink_param_length;
assign icmp_rx_depacketizer_sink_param_protocol = icmp_rx_sink_sink_param_protocol;
assign icmp_rx_depacketizer_sink_param_ip_address = icmp_rx_sink_sink_param_ip_address;
assign icmp_rx_source_source_valid = (icmp_rx_depacketizer_source_valid & icmp_rx_is_ongoing0);
assign icmp_rx_depacketizer_source_ready = (icmp_rx_is_ongoing1 | (icmp_rx_is_ongoing2 & icmp_rx_source_source_ready));
assign icmp_rx_source_source_payload_data = icmp_rx_depacketizer_source_payload_data;
assign icmp_rx_source_source_param_checksum = icmp_rx_depacketizer_source_param_checksum;
assign icmp_rx_source_source_param_code = icmp_rx_depacketizer_source_param_code;
assign icmp_rx_source_source_param_msgtype = icmp_rx_depacketizer_source_param_msgtype;
assign icmp_rx_source_source_param_quench = icmp_rx_depacketizer_source_param_quench;
assign icmp_rx_source_source_param_ip_address = icmp_rx_ip_address;
assign icmp_rx_source_source_param_length = (icmp_rx_length - 4'd8);
assign icmp_rx_depacketizer_header = icmp_rx_depacketizer_sr;
assign icmp_rx_depacketizer_source_param_checksum = {rhs_slice_proxy63[7:0], rhs_slice_proxy62[15:8]};
assign icmp_rx_depacketizer_source_param_code = {rhs_slice_proxy64[7:0]};
assign icmp_rx_depacketizer_source_param_msgtype = {rhs_slice_proxy65[7:0]};
assign icmp_rx_depacketizer_source_param_quench = {rhs_slice_proxy69[7:0], rhs_slice_proxy68[15:8], rhs_slice_proxy67[23:16], rhs_slice_proxy66[31:24]};
assign icmp_rx_depacketizer_sink_be = icmp_rx_depacketizer_sink_payload_be;
assign icmp_rx_depacketizer_source_payload_be = icmp_rx_depacketizer_source_be;
assign icmp_rx_depacketizer_source_payload_error = icmp_rx_depacketizer_sink_payload_error;

// synthesis translate_off
reg dummy_d_52;
// synthesis translate_on
always @(*) begin
	icmp_rx_depacketizer_sink_ready <= 1'd0;
	icmp_rx_depacketizer_source_valid <= 1'd0;
	icmp_rx_depacketizer_source_first <= 1'd0;
	icmp_rx_depacketizer_source_last <= 1'd0;
	icmp_rx_depacketizer_source_payload_data <= 8'd0;
	icmp_rx_depacketizer_sr_shift <= 1'd0;
	icmp_rx_depacketizer_source_be <= 1'd1;
	liteethicmp_liteethicmpdepacketizer_next_state <= 2'd0;
	icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value0 <= 3'd0;
	icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value_ce0 <= 1'd0;
	icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value1 <= 1'd0;
	icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value_ce1 <= 1'd0;
	icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value2 <= 1'd0;
	icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value_ce2 <= 1'd0;
	liteethicmp_liteethicmpdepacketizer_next_state <= liteethicmp_liteethicmpdepacketizer_state;
	case (liteethicmp_liteethicmpdepacketizer_state)
		1'd1: begin
			icmp_rx_depacketizer_sink_ready <= 1'd1;
			if (icmp_rx_depacketizer_sink_valid) begin
				icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value0 <= (icmp_rx_depacketizer_count + 1'd1);
				icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value_ce0 <= 1'd1;
				icmp_rx_depacketizer_sr_shift <= 1'd1;
				if ((icmp_rx_depacketizer_count == 3'd7)) begin
					icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value2 <= 1'd1;
					icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value_ce2 <= 1'd1;
					liteethicmp_liteethicmpdepacketizer_next_state <= 2'd2;
					icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value0 <= (icmp_rx_depacketizer_count + 1'd1);
					icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value_ce0 <= 1'd1;
				end
				if (icmp_rx_depacketizer_sink_last) begin
					liteethicmp_liteethicmpdepacketizer_next_state <= 1'd0;
				end
			end
		end
		2'd2: begin
			icmp_rx_depacketizer_source_valid <= icmp_rx_depacketizer_sink_valid;
			icmp_rx_depacketizer_source_first <= icmp_rx_depacketizer_data_copy_first;
			icmp_rx_depacketizer_source_last <= icmp_rx_depacketizer_sink_last;
			icmp_rx_depacketizer_sink_ready <= icmp_rx_depacketizer_source_ready;
			icmp_rx_depacketizer_source_payload_data <= icmp_rx_depacketizer_sink_payload_data;
			icmp_rx_depacketizer_source_be <= icmp_rx_depacketizer_sink_be;
			if ((icmp_rx_depacketizer_source_valid & icmp_rx_depacketizer_source_ready)) begin
				icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value2 <= 1'd0;
				icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value_ce2 <= 1'd1;
				icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value1 <= 1'd0;
				icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value_ce1 <= 1'd1;
				if (icmp_rx_depacketizer_source_last) begin
					liteethicmp_liteethicmpdepacketizer_next_state <= 1'd0;
				end
			end
		end
		default: begin
			icmp_rx_depacketizer_sink_ready <= 1'd1;
			icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value0 <= 1'd1;
			icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value_ce0 <= 1'd1;
			if (icmp_rx_depacketizer_sink_valid) begin
				icmp_rx_depacketizer_sr_shift <= 1'd1;
				icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value1 <= 1'd1;
				icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value_ce1 <= 1'd1;
				icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value2 <= 1'd1;
				icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value_ce2 <= 1'd1;
				if (1'd0) begin
					liteethicmp_liteethicmpdepacketizer_next_state <= 2'd2;
				end else begin
					liteethicmp_liteethicmpdepacketizer_next_state <= 1'd1;
				end
				if (($signed({1'd0, icmp_rx_depacketizer_sink_last}) & 1'sd1)) begin
					liteethicmp_liteethicmpdepacketizer_next_state <= 1'd0;
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_52 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_53;
// synthesis translate_on
always @(*) begin
	icmp_rx_source_source_last <= 1'd0;
	icmp_rx_source_source_payload_be <= 1'd0;
	icmp_rx_source_source_payload_error <= 1'd0;
	icmp_rx_is_ongoing0 <= 1'd0;
	icmp_rx_is_ongoing1 <= 1'd0;
	icmp_rx_is_ongoing2 <= 1'd0;
	liteethicmp_fsm_next_state <= 2'd0;
	icmp_rx_count_liteethicmp_fsm_next_value <= 17'd0;
	icmp_rx_count_liteethicmp_fsm_next_value_ce <= 1'd0;
	icmp_rx_source_source_last <= icmp_rx_depacketizer_source_last;
	icmp_rx_source_source_payload_be <= icmp_rx_depacketizer_source_payload_be;
	icmp_rx_source_source_payload_error <= icmp_rx_depacketizer_source_payload_error;
	liteethicmp_fsm_next_state <= liteethicmp_fsm_state;
	case (liteethicmp_fsm_state)
		1'd0: begin
			icmp_rx_is_ongoing0 <= 1'd1;
			icmp_rx_is_ongoing2 <= 1'd1;
			icmp_rx_source_source_last <= (icmp_rx_depacketizer_source_last | (icmp_rx_count >= icmp_rx_source_source_param_length));
			if ((icmp_rx_count >= icmp_rx_source_source_param_length)) begin
				icmp_rx_source_source_payload_be <= (icmp_rx_depacketizer_source_payload_be & {((((icmp_rx_source_source_param_length & 1'd0) == 1'd0) ? 1'd1 : (icmp_rx_source_source_param_length & 1'd0)) > 1'd0)});
			end
			if ((icmp_rx_depacketizer_source_last & (((icmp_rx_count - 1'd1) + (1'd0 + icmp_rx_depacketizer_source_payload_be)) < icmp_rx_source_source_param_length))) begin
				icmp_rx_source_source_payload_error <= icmp_rx_source_source_payload_be;
			end
			if ((icmp_rx_source_source_valid & icmp_rx_source_source_ready)) begin
				icmp_rx_count_liteethicmp_fsm_next_value <= (icmp_rx_count + 1'd1);
				icmp_rx_count_liteethicmp_fsm_next_value_ce <= 1'd1;
				if (icmp_rx_depacketizer_source_last) begin
					liteethicmp_fsm_next_state <= 2'd2;
				end else begin
					if (icmp_rx_source_source_last) begin
						liteethicmp_fsm_next_state <= 1'd1;
					end
				end
			end
		end
		1'd1: begin
			icmp_rx_is_ongoing1 <= 1'd1;
			if (((icmp_rx_depacketizer_source_valid & icmp_rx_depacketizer_source_last) & icmp_rx_depacketizer_source_ready)) begin
				liteethicmp_fsm_next_state <= 2'd2;
			end
		end
		default: begin
			icmp_rx_count_liteethicmp_fsm_next_value <= 1'd1;
			icmp_rx_count_liteethicmp_fsm_next_value_ce <= 1'd1;
			if (icmp_rx_depacketizer_source_valid) begin
				liteethicmp_fsm_next_state <= 1'd1;
				if (((icmp_rx_protocol == 1'd1) & (icmp_rx_length > 4'd8))) begin
					if (((icmp_rx_depacketizer_source_param_msgtype == 4'd8) & (icmp_rx_depacketizer_source_param_code == 1'd0))) begin
						liteethicmp_fsm_next_state <= 1'd0;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_53 <= dummy_s;
// synthesis translate_on
end
assign icmp_echo_buffer_sink_first = icmp_echo_sink_sink_first;
assign icmp_echo_buffer_sink_last = icmp_echo_sink_sink_last;
assign icmp_echo_buffer_sink_payload_data = icmp_echo_sink_sink_payload_data;
assign icmp_echo_buffer_sink_payload_be = icmp_echo_sink_sink_payload_be;
assign icmp_echo_buffer_sink_payload_error = icmp_echo_sink_sink_payload_error;
assign icmp_echo_buffer_sink_param_checksum = icmp_echo_sink_sink_param_checksum;
assign icmp_echo_buffer_sink_param_code = icmp_echo_sink_sink_param_code;
assign icmp_echo_buffer_sink_param_msgtype = icmp_echo_sink_sink_param_msgtype;
assign icmp_echo_buffer_sink_param_quench = icmp_echo_sink_sink_param_quench;
assign icmp_echo_buffer_sink_param_ip_address = icmp_echo_sink_sink_param_ip_address;
assign icmp_echo_buffer_sink_param_length = icmp_echo_sink_sink_param_length;
assign icmp_echo_buffer_sink_valid = (icmp_echo_sink_sink_valid & ((~icmp_echo_busy) | icmp_echo_receiving));
assign icmp_echo_sink_sink_ready = ((~icmp_echo_busy) | icmp_echo_receiving);
assign icmp_echo_size = (icmp_echo_count + (1'd0 + icmp_echo_sink_sink_payload_be));
assign icmp_echo_buffer_discard = (((((icmp_echo_sink_sink_param_length == 1'd0) | (icmp_echo_sink_sink_param_length > 8'd128)) | (icmp_echo_size > 8'd128)) | ((icmp_echo_sink_sink_payload_error & icmp_echo_sink_sink_payload_be) != 1'd0)) | (icmp_echo_sink_sink_last & (icmp_echo_size != icmp_echo_sink_sink_param_length)));
assign icmp_echo_source_source_valid = icmp_echo_buffer_source_valid;
assign icmp_echo_buffer_source_ready = icmp_echo_source_source_ready;
assign icmp_echo_source_source_first = icmp_echo_buffer_source_first;
assign icmp_echo_source_source_last = icmp_echo_buffer_source_last;
assign icmp_echo_source_source_payload_data = icmp_echo_buffer_source_payload_data;
assign icmp_echo_source_source_payload_be = icmp_echo_buffer_source_payload_be;
assign icmp_echo_source_source_payload_error = icmp_echo_buffer_source_payload_error;
assign icmp_echo_source_source_param_code = icmp_echo_buffer_source_param_code;
assign icmp_echo_source_source_param_quench = icmp_echo_buffer_source_param_quench;
assign icmp_echo_source_source_param_ip_address = icmp_echo_buffer_source_param_ip_address;
assign icmp_echo_source_source_param_length = icmp_echo_buffer_source_param_length;

// synthesis translate_off
reg dummy_d_54;
// synthesis translate_on
always @(*) begin
	icmp_echo_source_source_param_msgtype <= 8'd0;
	icmp_echo_source_source_param_msgtype <= icmp_echo_buffer_source_param_msgtype;
	icmp_echo_source_source_param_msgtype <= 1'd0;
// synthesis translate_off
	dummy_d_54 <= dummy_s;
// synthesis translate_on
end
assign icmp_echo_source_source_param_checksum = ((icmp_echo_buffer_source_param_checksum + 12'd2048) + (icmp_echo_buffer_source_param_checksum >= 16'd63488));
assign icmp_echo_buffer_used = (icmp_echo_buffer_wr_ptr - (icmp_echo_buffer_rd_ptr - icmp_echo_buffer_out_valid));
assign icmp_echo_buffer_full = (icmp_echo_buffer_used == 8'd128);
assign icmp_echo_buffer_sink_ready = 1'd1;
assign icmp_echo_buffer_wr_port_adr = icmp_echo_buffer_wr_ptr[6:0];
assign icmp_echo_buffer_wr_port_dat_w = {icmp_echo_buffer_sink_last, icmp_echo_buffer_sink_first, {icmp_echo_buffer_sink_payload_error, icmp_echo_buffer_sink_payload_be, icmp_echo_buffer_sink_payload_data}};
assign icmp_echo_buffer_wr_port_we = ((icmp_echo_buffer_sink_valid & (~icmp_echo_buffer_overflow)) & (~icmp_echo_buffer_full));
assign icmp_echo_buffer_commit = ((((((icmp_echo_buffer_sink_valid & icmp_echo_buffer_sink_last) & (~icmp_echo_buffer_overflow)) & (~icmp_echo_buffer_full)) & (~icmp_echo_buffer_invalid)) & (~icmp_echo_buffer_discard)) & icmp_echo_buffer_sink_sink_ready);
assign icmp_echo_buffer_sink_sink_param_checksum = icmp_echo_buffer_sink_param_checksum;
assign icmp_echo_buffer_sink_sink_param_code = icmp_echo_buffer_sink_param_code;
assign icmp_echo_buffer_sink_sink_param_msgtype = icmp_echo_buffer_sink_param_msgtype;
assign icmp_echo_buffer_sink_sink_param_quench = icmp_echo_buffer_sink_param_quench;
assign icmp_echo_buffer_sink_sink_param_ip_address = icmp_echo_buffer_sink_param_ip_address;
assign icmp_echo_buffer_sink_sink_param_length = icmp_echo_buffer_sink_param_length;
assign icmp_echo_buffer_sink_sink_valid = icmp_echo_buffer_commit;
assign icmp_echo_buffer_sink_sink_last = 1'd1;
assign icmp_echo_buffer_drop = ((icmp_echo_buffer_sink_valid & icmp_echo_buffer_sink_last) & (~icmp_echo_buffer_commit));
assign icmp_echo_buffer_fetch = ((icmp_echo_buffer_rd_ptr != icmp_echo_buffer_commit_ptr) & ((~icmp_echo_buffer_out_valid) | (icmp_echo_buffer_source_valid & icmp_echo_buffer_source_ready)));
assign icmp_echo_buffer_rd_port_adr = icmp_echo_buffer_rd_ptr[6:0];
assign icmp_echo_buffer_rd_port_re = icmp_echo_buffer_fetch;
assign icmp_echo_buffer_source_param_checksum = icmp_echo_buffer_source_source_param_checksum;
assign icmp_echo_buffer_source_param_code = icmp_echo_buffer_source_source_param_code;
assign icmp_echo_buffer_source_param_msgtype = icmp_echo_buffer_source_source_param_msgtype;
assign icmp_echo_buffer_source_param_quench = icmp_echo_buffer_source_source_param_quench;
assign icmp_echo_buffer_source_param_ip_address = icmp_echo_buffer_source_source_param_ip_address;
assign icmp_echo_buffer_source_param_length = icmp_echo_buffer_source_source_param_length;
assign {icmp_echo_buffer_source_payload_error, icmp_echo_buffer_source_payload_be, icmp_echo_buffer_source_payload_data} = icmp_echo_buffer_rd_port_dat_r[9:0];
assign icmp_echo_buffer_source_first = icmp_echo_buffer_rd_port_dat_r[10];
assign icmp_echo_buffer_source_last = icmp_echo_buffer_rd_port_dat_r[11];
assign icmp_echo_buffer_source_valid = (icmp_echo_buffer_out_valid & icmp_echo_buffer_source_source_valid);
assign icmp_echo_buffer_source_source_ready = ((icmp_echo_buffer_source_valid & icmp_echo_buffer_source_last) & icmp_echo_buffer_source_ready);
assign icmp_echo_buffer_level = icmp_echo_buffer_source_source_valid;
assign icmp_echo_buffer_pipe_valid_sink_ready = ((~icmp_echo_buffer_pipe_valid_source_valid) | icmp_echo_buffer_pipe_valid_source_ready);
assign icmp_echo_buffer_pipe_valid_sink_valid = icmp_echo_buffer_sink_sink_valid;
assign icmp_echo_buffer_sink_sink_ready = icmp_echo_buffer_pipe_valid_sink_ready;
assign icmp_echo_buffer_pipe_valid_sink_first = icmp_echo_buffer_sink_sink_first;
assign icmp_echo_buffer_pipe_valid_sink_last = icmp_echo_buffer_sink_sink_last;
assign icmp_echo_buffer_pipe_valid_sink_param_checksum = icmp_echo_buffer_sink_sink_param_checksum;
assign icmp_echo_buffer_pipe_valid_sink_param_code = icmp_echo_buffer_sink_sink_param_code;
assign icmp_echo_buffer_pipe_valid_sink_param_msgtype = icmp_echo_buffer_sink_sink_param_msgtype;
assign icmp_echo_buffer_pipe_valid_sink_param_quench = icmp_echo_buffer_sink_sink_param_quench;
assign icmp_echo_buffer_pipe_valid_sink_param_ip_address = icmp_echo_buffer_sink_sink_param_ip_address;
assign icmp_echo_buffer_pipe_valid_sink_param_length = icmp_echo_buffer_sink_sink_param_length;
assign icmp_echo_buffer_source_source_valid = icmp_echo_buffer_pipe_valid_source_valid;
assign icmp_echo_buffer_pipe_valid_source_ready = icmp_echo_buffer_source_source_ready;
assign icmp_echo_buffer_source_source_first = icmp_echo_buffer_pipe_valid_source_first;
assign icmp_echo_buffer_source_source_last = icmp_echo_buffer_pipe_valid_source_last;
assign icmp_echo_buffer_source_source_param_checksum = icmp_echo_buffer_pipe_valid_source_param_checksum;
assign icmp_echo_buffer_source_source_param_code = icmp_echo_buffer_pipe_valid_source_param_code;
assign icmp_echo_buffer_source_source_param_msgtype = icmp_echo_buffer_pipe_valid_source_param_msgtype;
assign icmp_echo_buffer_source_source_param_quench = icmp_echo_buffer_pipe_valid_source_param_quench;
assign icmp_echo_buffer_source_source_param_ip_address = icmp_echo_buffer_pipe_valid_source_param_ip_address;
assign icmp_echo_buffer_source_source_param_length = icmp_echo_buffer_pipe_valid_source_param_length;
assign ip_port_sink_valid = tx_source_source_valid;
assign tx_source_source_ready = ip_port_sink_ready;
assign ip_port_sink_first = tx_source_source_first;
assign ip_port_sink_last = tx_source_source_last;
assign ip_port_sink_payload_data = tx_source_source_payload_data;
assign ip_port_sink_payload_be = tx_source_source_payload_be;
assign ip_port_sink_payload_error = tx_source_source_payload_error;
assign ip_port_sink_param_length = tx_source_source_param_length;
assign ip_port_sink_param_protocol = tx_source_source_param_protocol;
assign ip_port_sink_param_ip_address = tx_source_source_param_ip_address;
assign rx_sink_sink_valid = ip_port_source_valid;
assign ip_port_source_ready = rx_sink_sink_ready;
assign rx_sink_sink_first = ip_port_source_first;
assign rx_sink_sink_last = ip_port_source_last;
assign rx_sink_sink_payload_data = ip_port_source_payload_data;
assign rx_sink_sink_payload_be = ip_port_source_payload_be;
assign rx_sink_sink_payload_error = ip_port_source_payload_error;
assign rx_sink_sink_param_length = ip_port_source_param_length;
assign rx_sink_sink_param_protocol = ip_port_source_param_protocol;
assign rx_sink_sink_param_ip_address = ip_port_source_param_ip_address;
assign tx_sink_sink_valid = crossbar_source_valid;
assign crossbar_source_ready = tx_sink_sink_ready;
assign tx_sink_sink_first = crossbar_source_first;
assign tx_sink_sink_last = crossbar_source_last;
assign tx_sink_sink_payload_data = crossbar_source_payload_data;
assign tx_sink_sink_payload_be = crossbar_source_payload_be;
assign tx_sink_sink_payload_error = crossbar_source_payload_error;
assign tx_sink_sink_param_src_port = crossbar_source_param_src_port;
assign tx_sink_sink_param_dst_port = crossbar_source_param_dst_port;
assign tx_sink_sink_param_ip_address = crossbar_source_param_ip_address;
assign tx_sink_sink_param_length = crossbar_source_param_length;
assign crossbar_sink_valid = rx_source_source_valid;
assign rx_source_source_ready = crossbar_sink_ready;
assign crossbar_sink_first = rx_source_source_first;
assign crossbar_sink_last = rx_source_source_last;
assign crossbar_sink_payload_data = rx_source_source_payload_data;
assign crossbar_sink_payload_be = rx_source_source_payload_be;
assign crossbar_sink_payload_error = rx_source_source_payload_error;
assign crossbar_sink_param_src_port = rx_source_source_param_src_port;
assign crossbar_sink_param_dst_port = rx_source_source_param_dst_port;
assign crossbar_sink_param_ip_address = rx_source_source_param_ip_address;
assign crossbar_sink_param_length = rx_source_source_param_length;
assign tx_packetizer_sink_valid = tx_sink_sink_valid;
assign tx_sink_sink_ready = tx_packetizer_sink_ready;
assign tx_packetizer_sink_last = tx_sink_sink_last;
assign tx_packetizer_sink_payload_data = tx_sink_sink_payload_data;
assign tx_packetizer_sink_payload_be = tx_sink_sink_payload_be;
assign tx_packetizer_sink_param_src_port = tx_sink_sink_param_src_port;
assign tx_packetizer_sink_param_dst_port = tx_sink_sink_param_dst_port;
assign tx_packetizer_sink_param_length = (tx_sink_sink_param_length + 4'd8);
assign tx_packetizer_sink_param_checksum = 1'd0;
assign tx_source_source_valid = tx_packetizer_source_valid;
assign tx_packetizer_source_ready = tx_source_source_ready;
assign tx_source_source_first = tx_packetizer_source_first;
assign tx_source_source_last = tx_packetizer_source_last;
assign tx_source_source_payload_data = tx_packetizer_source_payload_data;
assign tx_source_source_payload_be = tx_packetizer_source_payload_be;
assign tx_source_source_payload_error = tx_packetizer_source_payload_error;
assign tx_source_source_param_length = (tx_active ? tx_length : tx_packetizer_sink_param_length);
assign tx_source_source_param_protocol = 5'd17;
assign tx_source_source_param_ip_address = (tx_active ? tx_ip_address : tx_sink_sink_param_ip_address);

// synthesis translate_off
reg dummy_d_55;
// synthesis translate_on
always @(*) begin
	tx_packetizer_header <= 64'd0;
	tx_packetizer_header[63:48] <= {tx_packetizer_sink_param_checksum[7:0], tx_packetizer_sink_param_checksum[15:8]};
	tx_packetizer_header[31:16] <= {tx_packetizer_sink_param_dst_port[7:0], tx_packetizer_sink_param_dst_port[15:8]};
	tx_packetizer_header[47:32] <= {tx_packetizer_sink_param_length[7:0], tx_packetizer_sink_param_length[15:8]};
	tx_packetizer_header[15:0] <= {tx_packetizer_sink_param_src_port[7:0], tx_packetizer_sink_param_src_port[15:8]};
// synthesis translate_off
	dummy_d_55 <= dummy_s;
// synthesis translate_on
end
assign tx_packetizer_source_payload_be = tx_packetizer_source_be;

// synthesis translate_off
reg dummy_d_56;
// synthesis translate_on
always @(*) begin
	tx_packetizer_source_payload_error <= 1'd0;
	if (tx_packetizer_is_ongoing) begin
		tx_packetizer_source_payload_error <= tx_packetizer_sink_payload_error;
	end
// synthesis translate_off
	dummy_d_56 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_57;
// synthesis translate_on
always @(*) begin
	tx_packetizer_sink_ready <= 1'd0;
	tx_packetizer_source_valid <= 1'd0;
	tx_packetizer_source_first <= 1'd0;
	tx_packetizer_source_last <= 1'd0;
	tx_packetizer_source_payload_data <= 8'd0;
	tx_packetizer_source_be <= 1'd1;
	tx_packetizer_is_ongoing <= 1'd0;
	liteethudp_next_state <= 1'd0;
	tx_packetizer_count_liteethudp_next_value <= 3'd0;
	tx_packetizer_count_liteethudp_next_value_ce <= 1'd0;
	liteethudp_next_state <= liteethudp_state;
	case (liteethudp_state)
		1'd0: begin
			tx_packetizer_is_ongoing <= 1'd1;
			tx_packetizer_source_valid <= tx_packetizer_sink_valid;
			tx_packetizer_sink_ready <= tx_packetizer_source_ready;
			tx_packetizer_source_last <= tx_packetizer_sink_last;
			tx_packetizer_source_payload_data <= tx_packetizer_sink_payload_data;
			tx_packetizer_source_be <= tx_packetizer_sink_payload_be;
			if (((tx_packetizer_source_valid & tx_packetizer_source_ready) & tx_packetizer_source_last)) begin
				liteethudp_next_state <= 1'd1;
			end
		end
		default: begin
			tx_packetizer_sink_ready <= (tx_packetizer_count == 1'd0);
			case (tx_packetizer_count)
				1'd0: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[7:0];
				end
				1'd1: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[15:8];
				end
				2'd2: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[23:16];
				end
				2'd3: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[31:24];
				end
				3'd4: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[39:32];
				end
				3'd5: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[47:40];
				end
				3'd6: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[55:48];
				end
				3'd7: begin
					tx_packetizer_source_payload_data <= tx_packetizer_header[63:56];
				end
			endcase
			if ((tx_packetizer_count == 3'd7)) begin
				tx_packetizer_source_be <= 1'd1;
			end
			if ((tx_packetizer_sink_valid & 1'd1)) begin
				tx_packetizer_source_valid <= 1'd1;
				tx_packetizer_source_first <= (tx_packetizer_count == 1'd0);
				tx_packetizer_source_last <= ((tx_packetizer_count == 3'd7) & (tx_packetizer_sink_last & (tx_packetizer_sink_payload_be == 1'd0)));
				tx_packetizer_sink_ready <= (((tx_packetizer_count == 3'd7) & (tx_packetizer_sink_last & (tx_packetizer_sink_payload_be == 1'd0))) & tx_packetizer_source_ready);
				if (tx_packetizer_source_ready) begin
					if ((tx_packetizer_count == 3'd7)) begin
						tx_packetizer_count_liteethudp_next_value <= 1'd0;
						tx_packetizer_count_liteethudp_next_value_ce <= 1'd1;
						if ((~tx_packetizer_source_last)) begin
							liteethudp_next_state <= 1'd0;
						end
					end else begin
						tx_packetizer_count_liteethudp_next_value <= (tx_packetizer_count + 1'd1);
						tx_packetizer_count_liteethudp_next_value_ce <= 1'd1;
					end
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_57 <= dummy_s;
// synthesis translate_on
end
assign rx_depacketizer_sink_valid = rx_sink_sink_valid;
assign rx_sink_sink_ready = rx_depacketizer_sink_ready;
assign rx_depacketizer_sink_first = rx_sink_sink_first;
assign rx_depacketizer_sink_last = rx_sink_sink_last;
assign rx_depacketizer_sink_payload_data = rx_sink_sink_payload_data;
assign rx_depacketizer_sink_payload_be = rx_sink_sink_payload_be;
assign rx_depacketizer_sink_payload_error = rx_sink_sink_payload_error;
assign rx_depacketizer_sink_param_length = rx_sink_sink_param_length;
assign rx_depacketizer_sink_param_protocol = rx_sink_sink_param_protocol;
assign rx_depacketizer_sink_param_ip_address = rx_sink_sink_param_ip_address;
assign rx_source_source_payload_data = rx_depacketizer_source_payload_data;
assign rx_source_source_param_dst_port = rx_depacketizer_source_param_dst_port;
assign rx_source_source_param_src_port = rx_depacketizer_source_param_src_port;
assign rx_source_source_param_ip_address = rx_ip_address;
assign rx_source_source_param_length = (rx_depacketizer_source_param_length - 4'd8);
assign rx_accepted = (((rx_protocol == 5'd17) & (rx_depacketizer_source_param_length > 4'd8)) & (rx_depacketizer_source_param_length <= rx_length));
assign rx_receiving = rx_fsm_is_ongoing0;
assign rx_end_count = rx_count;
assign rx_source_source_valid = (rx_depacketizer_source_valid & rx_receiving);
assign rx_depacketizer_source_ready = (rx_fsm_is_ongoing1 | (rx_receiving & rx_source_source_ready));
assign rx_source_source_last = (rx_depacketizer_source_last | (rx_end_count >= rx_source_source_param_length));

// synthesis translate_off
reg dummy_d_58;
// synthesis translate_on
always @(*) begin
	rx_source_source_payload_be <= 1'd0;
	rx_source_source_payload_be <= rx_depacketizer_source_payload_be;
	if ((rx_end_count >= rx_source_source_param_length)) begin
		rx_source_source_payload_be <= (rx_depacketizer_source_payload_be & {((((rx_source_source_param_length & 1'd0) == 1'd0) ? 1'd1 : (rx_source_source_param_length & 1'd0)) > 1'd0)});
	end
// synthesis translate_off
	dummy_d_58 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_59;
// synthesis translate_on
always @(*) begin
	rx_source_source_payload_error <= 1'd0;
	rx_source_source_payload_error <= rx_depacketizer_source_payload_error;
	if ((rx_depacketizer_source_last & (((rx_end_count - 1'd1) + (1'd0 + rx_depacketizer_source_payload_be)) < rx_source_source_param_length))) begin
		rx_source_source_payload_error <= rx_source_source_payload_be;
	end
// synthesis translate_off
	dummy_d_59 <= dummy_s;
// synthesis translate_on
end
assign rx_depacketizer_header = rx_depacketizer_sr;
assign rx_depacketizer_source_param_checksum = {rhs_slice_proxy71[7:0], rhs_slice_proxy70[15:8]};
assign rx_depacketizer_source_param_dst_port = {rhs_slice_proxy73[7:0], rhs_slice_proxy72[15:8]};
assign rx_depacketizer_source_param_length = {rhs_slice_proxy75[7:0], rhs_slice_proxy74[15:8]};
assign rx_depacketizer_source_param_src_port = {rhs_slice_proxy77[7:0], rhs_slice_proxy76[15:8]};
assign rx_depacketizer_source_payload_error = rx_depacketizer_sink_payload_error;
assign rx_depacketizer_sr_shift = ((((rx_depacketizer_is_ongoing0 | rx_depacketizer_is_ongoing1) & rx_depacketizer_sink_valid) & rx_depacketizer_sink_ready) & (rx_depacketizer_skip_cnt >= 1'd0));

// synthesis translate_off
reg dummy_d_60;
// synthesis translate_on
always @(*) begin
	rx_depacketizer_sink_ready <= 1'd0;
	rx_depacketizer_source_valid <= 1'd0;
	rx_depacketizer_source_first <= 1'd0;
	rx_depacketizer_source_last <= 1'd0;
	rx_depacketizer_source_payload_data <= 8'd0;
	rx_depacketizer_source_payload_be <= 1'd0;
	rx_depacketizer_is_ongoing0 <= 1'd0;
	rx_depacketizer_is_ongoing1 <= 1'd0;
	liteethudp_liteethudpdepacketizer_next_state <= 2'd0;
	rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value <= 3'd0;
	rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value_ce <= 1'd0;
	rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value0 <= 1'd0;
	rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value_ce0 <= 1'd0;
	rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value1 <= 1'd0;
	rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value_ce1 <= 1'd0;
	rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value <= 1'd0;
	rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value_ce <= 1'd0;
	rx_depacketizer_source_payload_data <= rx_depacketizer_sink_payload_data;
	rx_depacketizer_source_payload_be <= rx_depacketizer_sink_payload_be;
	liteethudp_liteethudpdepacketizer_next_state <= liteethudp_liteethudpdepacketizer_state;
	case (liteethudp_liteethudpdepacketizer_state)
		1'd1: begin
			rx_depacketizer_is_ongoing1 <= 1'd1;
			rx_depacketizer_sink_ready <= 1'd1;
			if ((rx_depacketizer_sink_valid & rx_depacketizer_sink_ready)) begin
				rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value <= (rx_depacketizer_count + 1'd1);
				rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value_ce <= 1'd1;
				if ((rx_depacketizer_count == 3'd7)) begin
					rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value0 <= 1'd1;
					rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value_ce0 <= 1'd1;
					liteethudp_liteethudpdepacketizer_next_state <= 2'd2;
					rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value <= (rx_depacketizer_count + 1'd1);
					rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value_ce <= 1'd1;
				end
				if ((rx_depacketizer_sink_last & rx_depacketizer_sink_ready)) begin
					liteethudp_liteethudpdepacketizer_next_state <= 1'd0;
				end
			end
		end
		2'd2: begin
			if ((1'd0 & rx_depacketizer_send_dummy)) begin
				rx_depacketizer_source_valid <= 1'd1;
				rx_depacketizer_source_payload_data <= 1'd0;
				rx_depacketizer_source_first <= 1'd1;
				rx_depacketizer_source_last <= 1'd0;
				rx_depacketizer_source_payload_be <= 1'd1;
				rx_depacketizer_sink_ready <= 1'd0;
				if (rx_depacketizer_source_ready) begin
					rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value1 <= 1'd0;
					rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value_ce1 <= 1'd1;
				end
			end else begin
				rx_depacketizer_source_valid <= rx_depacketizer_sink_valid;
				rx_depacketizer_source_last <= rx_depacketizer_sink_last;
				rx_depacketizer_source_first <= (rx_depacketizer_data_copy_first & rx_depacketizer_sink_valid);
				rx_depacketizer_sink_ready <= rx_depacketizer_source_ready;
				if ((rx_depacketizer_source_valid & rx_depacketizer_source_ready)) begin
					rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value0 <= 1'd0;
					rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value_ce0 <= 1'd1;
					if (rx_depacketizer_source_last) begin
						rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value <= 1'd0;
						rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value_ce <= 1'd1;
						if (1'd0) begin
							rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value1 <= 1'd1;
							rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value_ce1 <= 1'd1;
						end
						liteethudp_liteethudpdepacketizer_next_state <= 1'd0;
					end
				end
			end
		end
		default: begin
			rx_depacketizer_is_ongoing0 <= 1'd1;
			rx_depacketizer_sink_ready <= 1'd1;
			rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value <= 1'd1;
			rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value_ce <= 1'd1;
			if ((rx_depacketizer_sink_valid & rx_depacketizer_sink_ready)) begin
				liteethudp_liteethudpdepacketizer_next_state <= 1'd1;
			end
		end
	endcase
// synthesis translate_off
	dummy_d_60 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_61;
// synthesis translate_on
always @(*) begin
	rx_fsm_is_ongoing0 <= 1'd0;
	rx_fsm_is_ongoing1 <= 1'd0;
	liteethudp_fsm_next_state <= 2'd0;
	rx_count_liteethudp_fsm_next_value <= 17'd0;
	rx_count_liteethudp_fsm_next_value_ce <= 1'd0;
	liteethudp_fsm_next_state <= liteethudp_fsm_state;
	case (liteethudp_fsm_state)
		1'd0: begin
			rx_fsm_is_ongoing0 <= 1'd1;
			if ((rx_source_source_valid & rx_source_source_ready)) begin
				rx_count_liteethudp_fsm_next_value <= (rx_end_count + 1'd1);
				rx_count_liteethudp_fsm_next_value_ce <= 1'd1;
				liteethudp_fsm_next_state <= 1'd0;
				if (rx_depacketizer_source_last) begin
					liteethudp_fsm_next_state <= 2'd2;
				end else begin
					if (rx_source_source_last) begin
						liteethudp_fsm_next_state <= 1'd1;
					end
				end
			end
		end
		1'd1: begin
			rx_fsm_is_ongoing1 <= 1'd1;
			if (((rx_depacketizer_source_valid & rx_depacketizer_source_last) & rx_depacketizer_source_ready)) begin
				liteethudp_fsm_next_state <= 2'd2;
			end
		end
		default: begin
			rx_count_liteethudp_fsm_next_value <= 1'd1;
			rx_count_liteethudp_fsm_next_value_ce <= 1'd1;
			if (rx_depacketizer_source_valid) begin
				liteethudp_fsm_next_state <= 1'd1;
				if (rx_accepted) begin
					liteethudp_fsm_next_state <= 1'd0;
				end
			end
		end
	endcase
// synthesis translate_off
	dummy_d_61 <= dummy_s;
// synthesis translate_on
end
assign tx_cdc_sink_valid = user_port_sink_valid;
assign user_port_sink_ready = tx_cdc_sink_ready;
assign tx_cdc_sink_first = user_port_sink_first;
assign tx_cdc_sink_last = user_port_sink_last;
assign tx_cdc_sink_payload_data = user_port_sink_payload_data;
assign tx_cdc_sink_payload_be = user_port_sink_payload_be;
assign tx_cdc_sink_payload_error = user_port_sink_payload_error;
assign tx_cdc_sink_param_src_port = user_port_sink_param_src_port;
assign tx_cdc_sink_param_dst_port = user_port_sink_param_dst_port;
assign tx_cdc_sink_param_ip_address = user_port_sink_param_ip_address;
assign tx_cdc_sink_param_length = user_port_sink_param_length;
assign tx_converter_sink_sink_valid = tx_cdc_source_valid;
assign tx_cdc_source_ready = tx_converter_sink_sink_ready;
assign tx_converter_sink_sink_first = tx_cdc_source_first;
assign tx_converter_sink_sink_last = tx_cdc_source_last;
assign tx_converter_sink_sink_payload_data = tx_cdc_source_payload_data;
assign tx_converter_sink_sink_payload_be = tx_cdc_source_payload_be;
assign tx_converter_sink_sink_payload_error = tx_cdc_source_payload_error;
assign tx_converter_sink_sink_param_src_port = tx_cdc_source_param_src_port;
assign tx_converter_sink_sink_param_dst_port = tx_cdc_source_param_dst_port;
assign tx_converter_sink_sink_param_ip_address = tx_cdc_source_param_ip_address;
assign tx_converter_sink_sink_param_length = tx_cdc_source_param_length;
assign internal_port_sink_valid = tx_converter_source_source_valid;
assign tx_converter_source_source_ready = internal_port_sink_ready;
assign internal_port_sink_first = tx_converter_source_source_first;
assign internal_port_sink_last = tx_converter_source_source_last;
assign internal_port_sink_payload_data = tx_converter_source_source_payload_data;
assign internal_port_sink_payload_be = tx_converter_source_source_payload_be;
assign internal_port_sink_payload_error = tx_converter_source_source_payload_error;
assign internal_port_sink_param_src_port = tx_converter_source_source_param_src_port;
assign internal_port_sink_param_dst_port = tx_converter_source_source_param_dst_port;
assign internal_port_sink_param_ip_address = tx_converter_source_source_param_ip_address;
assign internal_port_sink_param_length = tx_converter_source_source_param_length;
assign rx_converter_sink_sink_valid = internal_port_source_valid;
assign internal_port_source_ready = rx_converter_sink_sink_ready;
assign rx_converter_sink_sink_first = internal_port_source_first;
assign rx_converter_sink_sink_last = internal_port_source_last;
assign rx_converter_sink_sink_payload_data = internal_port_source_payload_data;
assign rx_converter_sink_sink_payload_be = internal_port_source_payload_be;
assign rx_converter_sink_sink_payload_error = internal_port_source_payload_error;
assign rx_converter_sink_sink_param_src_port = internal_port_source_param_src_port;
assign rx_converter_sink_sink_param_dst_port = internal_port_source_param_dst_port;
assign rx_converter_sink_sink_param_ip_address = internal_port_source_param_ip_address;
assign rx_converter_sink_sink_param_length = internal_port_source_param_length;
assign rx_cdc_sink_valid = rx_converter_source_source_valid;
assign rx_converter_source_source_ready = rx_cdc_sink_ready;
assign rx_cdc_sink_first = rx_converter_source_source_first;
assign rx_cdc_sink_last = rx_converter_source_source_last;
assign rx_cdc_sink_payload_data = rx_converter_source_source_payload_data;
assign rx_cdc_sink_payload_be = rx_converter_source_source_payload_be;
assign rx_cdc_sink_payload_error = rx_converter_source_source_payload_error;
assign rx_cdc_sink_param_src_port = rx_converter_source_source_param_src_port;
assign rx_cdc_sink_param_dst_port = rx_converter_source_source_param_dst_port;
assign rx_cdc_sink_param_ip_address = rx_converter_source_source_param_ip_address;
assign rx_cdc_sink_param_length = rx_converter_source_source_param_length;
assign user_port_source_valid = rx_cdc_source_valid;
assign rx_cdc_source_ready = user_port_source_ready;
assign user_port_source_first = rx_cdc_source_first;
assign user_port_source_last = rx_cdc_source_last;
assign user_port_source_payload_data = rx_cdc_source_payload_data;
assign user_port_source_payload_be = rx_cdc_source_payload_be;
assign user_port_source_payload_error = rx_cdc_source_payload_error;
assign user_port_source_param_src_port = rx_cdc_source_param_src_port;
assign user_port_source_param_dst_port = rx_cdc_source_param_dst_port;
assign user_port_source_param_ip_address = rx_cdc_source_param_ip_address;
assign user_port_source_param_length = rx_cdc_source_param_length;

// synthesis translate_off
reg dummy_d_62;
// synthesis translate_on
always @(*) begin
	liteethudp_sel0 <= 1'd0;
	if ((crossbar_sink_param_dst_port == 11'd1234)) begin
		liteethudp_sel0 <= 1'd1;
	end
// synthesis translate_off
	dummy_d_62 <= dummy_s;
// synthesis translate_on
end
assign tx_cdc_source_valid = tx_cdc_sink_valid;
assign tx_cdc_sink_ready = tx_cdc_source_ready;
assign tx_cdc_source_first = tx_cdc_sink_first;
assign tx_cdc_source_last = tx_cdc_sink_last;
assign tx_cdc_source_payload_data = tx_cdc_sink_payload_data;
assign tx_cdc_source_payload_be = tx_cdc_sink_payload_be;
assign tx_cdc_source_payload_error = tx_cdc_sink_payload_error;
assign tx_cdc_source_param_src_port = tx_cdc_sink_param_src_port;
assign tx_cdc_source_param_dst_port = tx_cdc_sink_param_dst_port;
assign tx_cdc_source_param_ip_address = tx_cdc_sink_param_ip_address;
assign tx_cdc_source_param_length = tx_cdc_sink_param_length;
assign tx_converter_converter_sink_valid = tx_converter_sink_sink_valid;
assign tx_converter_converter_sink_first = tx_converter_sink_sink_first;
assign tx_converter_converter_sink_last = tx_converter_sink_sink_last;
assign tx_converter_sink_sink_ready = tx_converter_converter_sink_ready;
assign tx_converter_converter_sink_payload_data = {tx_converter_sink_sink_payload_error, tx_converter_sink_sink_payload_be, tx_converter_sink_sink_payload_data};
assign tx_converter_source_source_valid = tx_converter_converter_source_valid;
assign tx_converter_source_source_first = tx_converter_converter_source_first;
assign tx_converter_source_source_last = tx_converter_converter_source_last;
assign tx_converter_converter_source_ready = tx_converter_source_source_ready;
assign {tx_converter_source_source_payload_error, tx_converter_source_source_payload_be, tx_converter_source_source_payload_data} = tx_converter_converter_source_payload_data;
assign tx_converter_source_source_param_src_port = tx_converter_sink_sink_param_src_port;
assign tx_converter_source_source_param_dst_port = tx_converter_sink_sink_param_dst_port;
assign tx_converter_source_source_param_ip_address = tx_converter_sink_sink_param_ip_address;
assign tx_converter_source_source_param_length = tx_converter_sink_sink_param_length;
assign tx_converter_converter_source_valid = tx_converter_converter_sink_valid;
assign tx_converter_converter_sink_ready = tx_converter_converter_source_ready;
assign tx_converter_converter_source_first = tx_converter_converter_sink_first;
assign tx_converter_converter_source_last = tx_converter_converter_sink_last;
assign tx_converter_converter_source_payload_data = tx_converter_converter_sink_payload_data;
assign tx_converter_converter_source_payload_valid_token_count = 1'd1;
assign rx_converter_converter_sink_valid = rx_converter_sink_sink_valid;
assign rx_converter_converter_sink_first = rx_converter_sink_sink_first;
assign rx_converter_converter_sink_last = rx_converter_sink_sink_last;
assign rx_converter_sink_sink_ready = rx_converter_converter_sink_ready;
assign rx_converter_converter_sink_payload_data = {rx_converter_sink_sink_payload_error, rx_converter_sink_sink_payload_be, rx_converter_sink_sink_payload_data};
assign rx_converter_source_source_valid = rx_converter_converter_source_valid;
assign rx_converter_source_source_first = rx_converter_converter_source_first;
assign rx_converter_source_source_last = rx_converter_converter_source_last;
assign rx_converter_converter_source_ready = rx_converter_source_source_ready;
assign {rx_converter_source_source_payload_error, rx_converter_source_source_payload_be, rx_converter_source_source_payload_data} = rx_converter_converter_source_payload_data;
assign rx_converter_source_source_param_src_port = rx_converter_sink_sink_param_src_port;
assign rx_converter_source_source_param_dst_port = rx_converter_sink_sink_param_dst_port;
assign rx_converter_source_source_param_ip_address = rx_converter_sink_sink_param_ip_address;
assign rx_converter_source_source_param_length = rx_converter_sink_sink_param_length;
assign rx_converter_converter_source_valid = rx_converter_converter_sink_valid;
assign rx_converter_converter_sink_ready = rx_converter_converter_source_ready;
assign rx_converter_converter_source_first = rx_converter_converter_sink_first;
assign rx_converter_converter_source_last = rx_converter_converter_sink_last;
assign rx_converter_converter_source_payload_data = rx_converter_converter_sink_payload_data;
assign rx_converter_converter_source_payload_valid_token_count = 1'd1;
assign rx_cdc_source_valid = rx_cdc_sink_valid;
assign rx_cdc_sink_ready = rx_cdc_source_ready;
assign rx_cdc_source_first = rx_cdc_sink_first;
assign rx_cdc_source_last = rx_cdc_sink_last;
assign rx_cdc_source_payload_data = rx_cdc_sink_payload_data;
assign rx_cdc_source_payload_be = rx_cdc_sink_payload_be;
assign rx_cdc_source_payload_error = rx_cdc_sink_payload_error;
assign rx_cdc_source_param_src_port = rx_cdc_sink_param_src_port;
assign rx_cdc_source_param_dst_port = rx_cdc_sink_param_dst_port;
assign rx_cdc_source_param_ip_address = rx_cdc_sink_param_ip_address;
assign rx_cdc_source_param_length = rx_cdc_sink_param_length;
assign crossbar_source_valid = internal_port_sink_valid;
assign internal_port_sink_ready = crossbar_source_ready;
assign crossbar_source_first = internal_port_sink_first;
assign crossbar_source_last = internal_port_sink_last;
assign crossbar_source_payload_data = internal_port_sink_payload_data;
assign crossbar_source_payload_be = internal_port_sink_payload_be;
assign crossbar_source_payload_error = internal_port_sink_payload_error;
assign crossbar_source_param_src_port = internal_port_sink_param_src_port;
assign crossbar_source_param_dst_port = internal_port_sink_param_dst_port;
assign crossbar_source_param_ip_address = internal_port_sink_param_ip_address;
assign crossbar_source_param_length = internal_port_sink_param_length;

// synthesis translate_off
reg dummy_d_63;
// synthesis translate_on
always @(*) begin
	liteethudp_sel1 <= 1'd0;
	if ((~liteethudp_sel_locked)) begin
		liteethudp_sel1 <= liteethudp_sel0;
	end else begin
		liteethudp_sel1 <= liteethudp_sel_ongoing;
	end
// synthesis translate_off
	dummy_d_63 <= dummy_s;
// synthesis translate_on
end

// synthesis translate_off
reg dummy_d_64;
// synthesis translate_on
always @(*) begin
	crossbar_sink_ready <= 1'd0;
	internal_port_source_valid <= 1'd0;
	internal_port_source_first <= 1'd0;
	internal_port_source_last <= 1'd0;
	internal_port_source_payload_data <= 8'd0;
	internal_port_source_payload_be <= 1'd0;
	internal_port_source_payload_error <= 1'd0;
	internal_port_source_param_src_port <= 16'd0;
	internal_port_source_param_dst_port <= 16'd0;
	internal_port_source_param_ip_address <= 32'd0;
	internal_port_source_param_length <= 16'd0;
	case (liteethudp_sel1)
		1'd1: begin
			internal_port_source_valid <= crossbar_sink_valid;
			crossbar_sink_ready <= internal_port_source_ready;
			internal_port_source_first <= crossbar_sink_first;
			internal_port_source_last <= crossbar_sink_last;
			internal_port_source_payload_data <= crossbar_sink_payload_data;
			internal_port_source_payload_be <= crossbar_sink_payload_be;
			internal_port_source_payload_error <= crossbar_sink_payload_error;
			internal_port_source_param_src_port <= crossbar_sink_param_src_port;
			internal_port_source_param_dst_port <= crossbar_sink_param_dst_port;
			internal_port_source_param_ip_address <= crossbar_sink_param_ip_address;
			internal_port_source_param_length <= crossbar_sink_param_length;
		end
		default: begin
			crossbar_sink_ready <= 1'd1;
		end
	endcase
// synthesis translate_off
	dummy_d_64 <= dummy_s;
// synthesis translate_on
end
assign liteethudp_last = ((crossbar_sink_valid & crossbar_sink_last) & crossbar_sink_ready);
assign liteethudp_ongoing0 = ((crossbar_sink_valid | liteethudp_ongoing1) & (~liteethudp_last));
assign echo_buffer_syncfifo_din = {echo_buffer_fifo_in_last, echo_buffer_fifo_in_first, echo_buffer_fifo_in_param_length, echo_buffer_fifo_in_param_ip_address, echo_buffer_fifo_in_param_dst_port, echo_buffer_fifo_in_param_src_port, echo_buffer_fifo_in_payload_error, echo_buffer_fifo_in_payload_be, echo_buffer_fifo_in_payload_data};
assign {echo_buffer_fifo_out_last, echo_buffer_fifo_out_first, echo_buffer_fifo_out_param_length, echo_buffer_fifo_out_param_ip_address, echo_buffer_fifo_out_param_dst_port, echo_buffer_fifo_out_param_src_port, echo_buffer_fifo_out_payload_error, echo_buffer_fifo_out_payload_be, echo_buffer_fifo_out_payload_data} = echo_buffer_syncfifo_dout;
assign echo_buffer_sink_ready = echo_buffer_syncfifo_writable;
assign echo_buffer_syncfifo_we = echo_buffer_sink_valid;
assign echo_buffer_fifo_in_first = echo_buffer_sink_first;
assign echo_buffer_fifo_in_last = echo_buffer_sink_last;
assign echo_buffer_fifo_in_payload_data = echo_buffer_sink_payload_data;
assign echo_buffer_fifo_in_payload_be = echo_buffer_sink_payload_be;
assign echo_buffer_fifo_in_payload_error = echo_buffer_sink_payload_error;
assign echo_buffer_fifo_in_param_src_port = echo_buffer_sink_param_src_port;
assign echo_buffer_fifo_in_param_dst_port = echo_buffer_sink_param_dst_port;
assign echo_buffer_fifo_in_param_ip_address = echo_buffer_sink_param_ip_address;
assign echo_buffer_fifo_in_param_length = echo_buffer_sink_param_length;
assign echo_buffer_source_valid = echo_buffer_readable;
assign echo_buffer_source_first = echo_buffer_fifo_out_first;
assign echo_buffer_source_last = echo_buffer_fifo_out_last;
assign echo_buffer_source_payload_data = echo_buffer_fifo_out_payload_data;
assign echo_buffer_source_payload_be = echo_buffer_fifo_out_payload_be;
assign echo_buffer_source_payload_error = echo_buffer_fifo_out_payload_error;
assign echo_buffer_source_param_src_port = echo_buffer_fifo_out_param_src_port;
assign echo_buffer_source_param_dst_port = echo_buffer_fifo_out_param_dst_port;
assign echo_buffer_source_param_ip_address = echo_buffer_fifo_out_param_ip_address;
assign echo_buffer_source_param_length = echo_buffer_fifo_out_param_length;
assign echo_buffer_re = echo_buffer_source_ready;
assign echo_buffer_syncfifo_re = (echo_buffer_syncfifo_readable & ((~echo_buffer_readable) | echo_buffer_re));
assign echo_buffer_level1 = (echo_buffer_level0 + echo_buffer_readable);

// synthesis translate_off
reg dummy_d_65;
// synthesis translate_on
always @(*) begin
	echo_buffer_wrport_adr <= 8'd0;
	if (echo_buffer_replace) begin
		echo_buffer_wrport_adr <= (echo_buffer_produce - 1'd1);
	end else begin
		echo_buffer_wrport_adr <= echo_buffer_produce;
	end
// synthesis translate_off
	dummy_d_65 <= dummy_s;
// synthesis translate_on
end
assign echo_buffer_wrport_dat_w = echo_buffer_syncfifo_din;
assign echo_buffer_wrport_we = (echo_buffer_syncfifo_we & (echo_buffer_syncfifo_writable | echo_buffer_replace));
assign echo_buffer_do_read = (echo_buffer_syncfifo_readable & echo_buffer_syncfifo_re);
assign echo_buffer_rdport_adr = echo_buffer_consume;
assign echo_buffer_syncfifo_dout = echo_buffer_rdport_dat_r;
assign echo_buffer_rdport_re = echo_buffer_do_read;
assign echo_buffer_syncfifo_writable = (echo_buffer_level0 != 9'd256);
assign echo_buffer_syncfifo_readable = (echo_buffer_level0 != 1'd0);
assign uarttx_busy = (uarttx_state != 1'd0);
assign uart_txd = ((uarttx_state == 1'd0) ? 1'd1 : ((uarttx_state == 1'd1) ? 1'd0 : ((uarttx_state == 2'd2) ? uarttx_shreg[0] : 1'd1)));
assign complexslicelowerer_slice_proxy = {mac_core_tx_crc_value, mac_core_tx_crc_sink_payload_data[7:0]};
assign rhs_slice_proxy0 = mac_depacketizer_header[111:96];
assign rhs_slice_proxy1 = mac_depacketizer_header[111:96];
assign rhs_slice_proxy2 = mac_depacketizer_header[95:48];
assign rhs_slice_proxy3 = mac_depacketizer_header[95:48];
assign rhs_slice_proxy4 = mac_depacketizer_header[95:48];
assign rhs_slice_proxy5 = mac_depacketizer_header[95:48];
assign rhs_slice_proxy6 = mac_depacketizer_header[95:48];
assign rhs_slice_proxy7 = mac_depacketizer_header[95:48];
assign rhs_slice_proxy8 = mac_depacketizer_header[47:0];
assign rhs_slice_proxy9 = mac_depacketizer_header[47:0];
assign rhs_slice_proxy10 = mac_depacketizer_header[47:0];
assign rhs_slice_proxy11 = mac_depacketizer_header[47:0];
assign rhs_slice_proxy12 = mac_depacketizer_header[47:0];
assign rhs_slice_proxy13 = mac_depacketizer_header[47:0];
assign rhs_slice_proxy14 = arp_rx_depacketizer_header[39:32];
assign rhs_slice_proxy15 = arp_rx_depacketizer_header[15:0];
assign rhs_slice_proxy16 = arp_rx_depacketizer_header[15:0];
assign rhs_slice_proxy17 = arp_rx_depacketizer_header[63:48];
assign rhs_slice_proxy18 = arp_rx_depacketizer_header[63:48];
assign rhs_slice_proxy19 = arp_rx_depacketizer_header[31:16];
assign rhs_slice_proxy20 = arp_rx_depacketizer_header[31:16];
assign rhs_slice_proxy21 = arp_rx_depacketizer_header[47:40];
assign rhs_slice_proxy22 = arp_rx_depacketizer_header[143:112];
assign rhs_slice_proxy23 = arp_rx_depacketizer_header[143:112];
assign rhs_slice_proxy24 = arp_rx_depacketizer_header[143:112];
assign rhs_slice_proxy25 = arp_rx_depacketizer_header[143:112];
assign rhs_slice_proxy26 = arp_rx_depacketizer_header[111:64];
assign rhs_slice_proxy27 = arp_rx_depacketizer_header[111:64];
assign rhs_slice_proxy28 = arp_rx_depacketizer_header[111:64];
assign rhs_slice_proxy29 = arp_rx_depacketizer_header[111:64];
assign rhs_slice_proxy30 = arp_rx_depacketizer_header[111:64];
assign rhs_slice_proxy31 = arp_rx_depacketizer_header[111:64];
assign rhs_slice_proxy32 = arp_rx_depacketizer_header[223:192];
assign rhs_slice_proxy33 = arp_rx_depacketizer_header[223:192];
assign rhs_slice_proxy34 = arp_rx_depacketizer_header[223:192];
assign rhs_slice_proxy35 = arp_rx_depacketizer_header[223:192];
assign rhs_slice_proxy36 = arp_rx_depacketizer_header[191:144];
assign rhs_slice_proxy37 = arp_rx_depacketizer_header[191:144];
assign rhs_slice_proxy38 = arp_rx_depacketizer_header[191:144];
assign rhs_slice_proxy39 = arp_rx_depacketizer_header[191:144];
assign rhs_slice_proxy40 = arp_rx_depacketizer_header[191:144];
assign rhs_slice_proxy41 = arp_rx_depacketizer_header[191:144];
assign rhs_slice_proxy42 = ip_rx_depacketizer_header[95:80];
assign rhs_slice_proxy43 = ip_rx_depacketizer_header[95:80];
assign rhs_slice_proxy44 = ip_rx_depacketizer_header[63:48];
assign rhs_slice_proxy45 = ip_rx_depacketizer_header[63:48];
assign rhs_slice_proxy46 = ip_rx_depacketizer_header[47:32];
assign rhs_slice_proxy47 = ip_rx_depacketizer_header[47:32];
assign rhs_slice_proxy48 = ip_rx_depacketizer_header[3:0];
assign rhs_slice_proxy49 = ip_rx_depacketizer_header[79:72];
assign rhs_slice_proxy50 = ip_rx_depacketizer_header[127:96];
assign rhs_slice_proxy51 = ip_rx_depacketizer_header[127:96];
assign rhs_slice_proxy52 = ip_rx_depacketizer_header[127:96];
assign rhs_slice_proxy53 = ip_rx_depacketizer_header[127:96];
assign rhs_slice_proxy54 = ip_rx_depacketizer_header[159:128];
assign rhs_slice_proxy55 = ip_rx_depacketizer_header[159:128];
assign rhs_slice_proxy56 = ip_rx_depacketizer_header[159:128];
assign rhs_slice_proxy57 = ip_rx_depacketizer_header[159:128];
assign rhs_slice_proxy58 = ip_rx_depacketizer_header[31:16];
assign rhs_slice_proxy59 = ip_rx_depacketizer_header[31:16];
assign rhs_slice_proxy60 = ip_rx_depacketizer_header[71:64];
assign rhs_slice_proxy61 = ip_rx_depacketizer_header[7:4];
assign rhs_slice_proxy62 = icmp_rx_depacketizer_header[31:16];
assign rhs_slice_proxy63 = icmp_rx_depacketizer_header[31:16];
assign rhs_slice_proxy64 = icmp_rx_depacketizer_header[15:8];
assign rhs_slice_proxy65 = icmp_rx_depacketizer_header[7:0];
assign rhs_slice_proxy66 = icmp_rx_depacketizer_header[63:32];
assign rhs_slice_proxy67 = icmp_rx_depacketizer_header[63:32];
assign rhs_slice_proxy68 = icmp_rx_depacketizer_header[63:32];
assign rhs_slice_proxy69 = icmp_rx_depacketizer_header[63:32];
assign rhs_slice_proxy70 = rx_depacketizer_header[63:48];
assign rhs_slice_proxy71 = rx_depacketizer_header[63:48];
assign rhs_slice_proxy72 = rx_depacketizer_header[31:16];
assign rhs_slice_proxy73 = rx_depacketizer_header[31:16];
assign rhs_slice_proxy74 = rx_depacketizer_header[47:32];
assign rhs_slice_proxy75 = rx_depacketizer_header[47:32];
assign rhs_slice_proxy76 = rx_depacketizer_header[15:0];
assign rhs_slice_proxy77 = rx_depacketizer_header[15:0];

// synthesis translate_off
reg dummy_d_66;
// synthesis translate_on
always @(*) begin
	self <= 8'd0;
	case (idx)
		1'd0: begin
			self <= value0;
		end
		1'd1: begin
			self <= value1;
		end
		2'd2: begin
			self <= value2;
		end
		2'd3: begin
			self <= value3;
		end
		3'd4: begin
			self <= value4;
		end
		3'd5: begin
			self <= value5;
		end
		3'd6: begin
			self <= value6;
		end
		default: begin
			self <= value7;
		end
	endcase
// synthesis translate_off
	dummy_d_66 <= dummy_s;
// synthesis translate_on
end
assign mac_core_pulsesynchronizer0_toggle_o = multiregimpl01;
assign mac_core_pulsesynchronizer1_toggle_o = multiregimpl11;

always @(posedge eth_rx_clk) begin
	if ((phy_source_valid & phy_source_last)) begin
		hold0 <= 8'd128;
	end else begin
		if ((hold0 != 1'd0)) begin
			hold0 <= (hold0 - 1'd1);
		end
	end
	phy_rmiirxshim_crs_dv_d <= phy_rmiirxshim_crs_dv;
	if (phy_rmiirxshim_crs_first) begin
		phy_rmiirxshim_crs_run <= 1'd1;
	end
	if (phy_rmiirxshim_crs_last) begin
		phy_rmiirxshim_crs_run <= 1'd0;
	end
	phy_rmiirxshim_d0 <= phy_rmiirxshim_rx_data;
	phy_rmiirxshim_d1 <= phy_rmiirxshim_d0;
	phy_rmiirxshim_e0 <= phy_rmiirxshim_cap;
	phy_rmiirxshim_e1 <= phy_rmiirxshim_e0;
	if ((phy_rmiirxshim_e1 & (phy_rmiirxshim_mode == 1'd0))) begin
		phy_rmiirxshim_sreg <= {phy_rmiirxshim_sreg[5:0], phy_rmiirxshim_d1};
	end
	if ((phy_rmiirxshim_mode == 1'd0)) begin
		if (phy_rmiirxshim_sfd_hit) begin
			phy_rmiirxshim_mode <= 1'd1;
			phy_rmiirxshim_pre_cnt <= 1'd0;
			phy_rmiirxshim_bits <= 1'd0;
		end
	end else begin
		if ((phy_rmiirxshim_mode == 1'd1)) begin
			if (phy_rmiirxshim_consume) begin
				if ((phy_rmiirxshim_pre_cnt == 3'd7)) begin
					phy_rmiirxshim_mode <= 2'd2;
				end else begin
					phy_rmiirxshim_pre_cnt <= (phy_rmiirxshim_pre_cnt + 1'd1);
				end
			end
		end
	end
	if (((phy_rmiirxshim_mode == 2'd2) & phy_rmiirxshim_consume)) begin
		if (phy_rmiirxshim_pushing) begin
			phy_rmiirxshim_f0_valid <= 1'd1;
			phy_rmiirxshim_f0_data <= phy_rmiirxshim_f1_data;
			phy_rmiirxshim_f0_last <= phy_rmiirxshim_f1_last;
			phy_rmiirxshim_f1_valid <= 1'd1;
			phy_rmiirxshim_f1_data <= phy_rmiirxshim_new_byte;
			phy_rmiirxshim_f1_last <= phy_rmiirxshim_crs_last;
		end else begin
			phy_rmiirxshim_f0_valid <= phy_rmiirxshim_f1_valid;
			phy_rmiirxshim_f0_data <= phy_rmiirxshim_f1_data;
			phy_rmiirxshim_f0_last <= phy_rmiirxshim_f1_last;
			phy_rmiirxshim_f1_valid <= 1'd0;
		end
	end else begin
		if ((phy_rmiirxshim_pushing & (~(((phy_rmiirxshim_mode == 2'd2) & phy_rmiirxshim_f0_valid) & phy_rmiirxshim_f1_valid)))) begin
			if ((~phy_rmiirxshim_f0_valid)) begin
				phy_rmiirxshim_f0_valid <= 1'd1;
				phy_rmiirxshim_f0_data <= phy_rmiirxshim_new_byte;
				phy_rmiirxshim_f0_last <= phy_rmiirxshim_crs_last;
			end else begin
				if ((~phy_rmiirxshim_f1_valid)) begin
					phy_rmiirxshim_f1_valid <= 1'd1;
					phy_rmiirxshim_f1_data <= phy_rmiirxshim_new_byte;
					phy_rmiirxshim_f1_last <= phy_rmiirxshim_crs_last;
				end
			end
		end
	end
	if (((phy_rmiirxshim_mode != 1'd0) & phy_rmiirxshim_e1)) begin
		phy_rmiirxshim_shift <= phy_rmiirxshim_new_byte;
		if ((phy_rmiirxshim_bits == 2'd3)) begin
			phy_rmiirxshim_bits <= 1'd0;
		end else begin
			phy_rmiirxshim_bits <= (phy_rmiirxshim_bits + 1'd1);
		end
	end else begin
		if ((~phy_rmiirxshim_e1)) begin
			phy_rmiirxshim_bits <= 1'd0;
		end
	end
	if ((phy_rmiirxshim_consume & phy_rmiirxshim_source_last)) begin
		phy_rmiirxshim_mode <= 1'd0;
		phy_rmiirxshim_sreg <= 1'd0;
		phy_rmiirxshim_bits <= 1'd0;
		phy_rmiirxshim_f0_valid <= 1'd0;
		phy_rmiirxshim_f1_valid <= 1'd0;
	end
	if (((~mac_core_txdatapath_pipe_valid_source_valid) | mac_core_txdatapath_pipe_valid_source_ready)) begin
		mac_core_txdatapath_pipe_valid_source_valid <= mac_core_txdatapath_pipe_valid_sink_valid;
		mac_core_txdatapath_pipe_valid_source_first <= mac_core_txdatapath_pipe_valid_sink_first;
		mac_core_txdatapath_pipe_valid_source_last <= mac_core_txdatapath_pipe_valid_sink_last;
		mac_core_txdatapath_pipe_valid_source_payload_data <= mac_core_txdatapath_pipe_valid_sink_payload_data;
		mac_core_txdatapath_pipe_valid_source_payload_be <= mac_core_txdatapath_pipe_valid_sink_payload_be;
		mac_core_txdatapath_pipe_valid_source_payload_error <= mac_core_txdatapath_pipe_valid_sink_payload_error;
	end
	if (((~mac_core_rxdatapath_pipe_valid_source_valid) | mac_core_rxdatapath_pipe_valid_source_ready)) begin
		mac_core_rxdatapath_pipe_valid_source_valid <= mac_core_rxdatapath_pipe_valid_sink_valid;
		mac_core_rxdatapath_pipe_valid_source_first <= mac_core_rxdatapath_pipe_valid_sink_first;
		mac_core_rxdatapath_pipe_valid_source_last <= mac_core_rxdatapath_pipe_valid_sink_last;
		mac_core_rxdatapath_pipe_valid_source_payload_data <= mac_core_rxdatapath_pipe_valid_sink_payload_data;
		mac_core_rxdatapath_pipe_valid_source_payload_be <= mac_core_rxdatapath_pipe_valid_sink_payload_be;
		mac_core_rxdatapath_pipe_valid_source_payload_error <= mac_core_rxdatapath_pipe_valid_sink_payload_error;
	end
	if (eth_rx_rst) begin
		phy_rmiirxshim_crs_run <= 1'd0;
		phy_rmiirxshim_crs_dv_d <= 1'd0;
		phy_rmiirxshim_d0 <= 2'd0;
		phy_rmiirxshim_d1 <= 2'd0;
		phy_rmiirxshim_e0 <= 1'd0;
		phy_rmiirxshim_e1 <= 1'd0;
		phy_rmiirxshim_mode <= 2'd0;
		phy_rmiirxshim_f0_valid <= 1'd0;
		phy_rmiirxshim_f0_data <= 8'd0;
		phy_rmiirxshim_f0_last <= 1'd0;
		phy_rmiirxshim_f1_valid <= 1'd0;
		phy_rmiirxshim_f1_data <= 8'd0;
		phy_rmiirxshim_f1_last <= 1'd0;
		phy_rmiirxshim_sreg <= 8'd0;
		phy_rmiirxshim_pre_cnt <= 3'd0;
		phy_rmiirxshim_shift <= 8'd0;
		phy_rmiirxshim_bits <= 2'd0;
		mac_core_txdatapath_pipe_valid_source_valid <= 1'd0;
		mac_core_rxdatapath_pipe_valid_source_valid <= 1'd0;
		hold0 <= 8'd0;
	end
end

always @(posedge eth_tx_clk) begin
	if ((phy_rmiitxshim_sink_valid & phy_rmiitxshim_sink_ready)) begin
		phy_rmiitxshim_shift <= phy_rmiitxshim_sink_payload_data;
		phy_rmiitxshim_count <= 3'd4;
	end else begin
		if ((phy_rmiitxshim_count != 1'd0)) begin
			phy_rmiitxshim_shift <= (phy_rmiitxshim_shift >>> 2'd2);
			phy_rmiitxshim_count <= (phy_rmiitxshim_count - 1'd1);
		end
	end
	liteethmac_txdatapath_resetinserter_state <= liteethmac_txdatapath_resetinserter_next_state;
	if (mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value_ce) begin
		mac_core_tx_gap_counter <= mac_core_tx_gap_counter_liteethmaccore_clockdomainsrenamer3_next_value;
	end
	if (mac_core_tx_gap_reset) begin
		liteethmac_txdatapath_resetinserter_state <= 1'd0;
	end
	if (eth_tx_rst) begin
		phy_rmiitxshim_shift <= 8'd0;
		phy_rmiitxshim_count <= 3'd0;
		liteethmac_txdatapath_resetinserter_state <= 1'd0;
	end
end

always @(posedge sys_clk) begin
	if ((mac_core_rxdatapath_source_valid & mac_core_rxdatapath_source_last)) begin
		hold1 <= 8'd128;
	end else begin
		if ((hold1 != 1'd0)) begin
			hold1 <= (hold1 - 1'd1);
		end
	end
	if ((mac_depacketizer_sink_valid & mac_depacketizer_sink_last)) begin
		hold2 <= 8'd128;
	end else begin
		if ((hold2 != 1'd0)) begin
			hold2 <= (hold2 - 1'd1);
		end
	end
	if ((arp_table_sink_valid & arp_table_sink_payload_request)) begin
		hold3 <= 8'd128;
	end else begin
		if ((hold3 != 1'd0)) begin
			hold3 <= (hold3 - 1'd1);
		end
	end
	if (arp_table_source_valid) begin
		hold4 <= 8'd128;
	end else begin
		if ((hold4 != 1'd0)) begin
			hold4 <= (hold4 - 1'd1);
		end
	end
	if ((phy_sink_valid & phy_sink_last)) begin
		hold5 <= 8'd128;
	end else begin
		if ((hold5 != 1'd0)) begin
			hold5 <= (hold5 - 1'd1);
		end
	end
	if (fire) begin
		timer <= 23'd8388607;
	end else begin
		timer <= (timer - 1'd1);
	end
	fire <= (timer == 1'd0);
	if (fire) begin
		dumping <= 1'd1;
		idx <= 1'd0;
	end else begin
		if ((dumping & (~uarttx_busy))) begin
			if ((idx == 3'd7)) begin
				dumping <= 1'd0;
			end else begin
				idx <= (idx + 1'd1);
			end
		end
	end
	liteethmac_txdatapath_liteethmacpaddinginserter_state <= liteethmac_txdatapath_liteethmacpaddinginserter_next_state;
	if (mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value_ce) begin
		mac_core_tx_padding_counter <= mac_core_tx_padding_counter_liteethmaccore_clockdomainsrenamer0_next_value;
	end
	if (mac_core_tx_crc_is_ongoing0) begin
		mac_core_tx_crc_cnt <= 2'd3;
	end else begin
		if ((mac_core_tx_crc_is_ongoing1 & (~mac_core_tx_crc_cnt_done))) begin
			mac_core_tx_crc_cnt <= (mac_core_tx_crc_cnt - mac_core_tx_crc_source_ready);
		end
	end
	if (mac_core_tx_crc_ce) begin
		mac_core_tx_crc_reg <= mac_core_tx_crc_crc_next;
	end
	if (mac_core_tx_crc_reset) begin
		mac_core_tx_crc_reg <= 32'd4294967295;
	end
	liteethmac_txdatapath_bufferizeendpoints_state <= liteethmac_txdatapath_bufferizeendpoints_next_state;
	if (mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value_ce0) begin
		mac_core_tx_crc_crc_packet <= mac_core_tx_crc_crc_packet_liteethmaccore_clockdomainsrenamer1_next_value0;
	end
	if (mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value_ce1) begin
		mac_core_tx_crc_be0 <= mac_core_tx_crc_be0_liteethmaccore_clockdomainsrenamer1_next_value1;
	end
	if (((~mac_core_tx_crc_pipe_valid_source_valid) | mac_core_tx_crc_pipe_valid_source_ready)) begin
		mac_core_tx_crc_pipe_valid_source_valid <= mac_core_tx_crc_pipe_valid_sink_valid;
		mac_core_tx_crc_pipe_valid_source_first <= mac_core_tx_crc_pipe_valid_sink_first;
		mac_core_tx_crc_pipe_valid_source_last <= mac_core_tx_crc_pipe_valid_sink_last;
		mac_core_tx_crc_pipe_valid_source_payload_data <= mac_core_tx_crc_pipe_valid_sink_payload_data;
		mac_core_tx_crc_pipe_valid_source_payload_be <= mac_core_tx_crc_pipe_valid_sink_payload_be;
		mac_core_tx_crc_pipe_valid_source_payload_error <= mac_core_tx_crc_pipe_valid_sink_payload_error;
	end
	liteethmac_txdatapath_liteethmacpreambleinserter_state <= liteethmac_txdatapath_liteethmacpreambleinserter_next_state;
	if (mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value_ce) begin
		mac_core_tx_preamble_count <= mac_core_tx_preamble_count_liteethmaccore_clockdomainsrenamer2_next_value;
	end
	if (mac_core_pulsesynchronizer0_o) begin
		mac_core_preamble_errors_status <= (mac_core_preamble_errors_status + 1'd1);
	end
	if (mac_core_pulsesynchronizer1_o) begin
		mac_core_crc_errors_status <= (mac_core_crc_errors_status + 1'd1);
	end
	liteethmac_rxdatapath_liteethmacpreamblechecker_state <= liteethmac_rxdatapath_liteethmacpreamblechecker_next_state;
	if (mac_core_pulsesynchronizer0_i) begin
		mac_core_pulsesynchronizer0_toggle_i <= (~mac_core_pulsesynchronizer0_toggle_i);
	end
	mac_core_pulsesynchronizer0_toggle_o_r <= mac_core_pulsesynchronizer0_toggle_o;
	if (mac_core_liteethmaccrc32checker_is_ongoing2) begin
		mac_core_liteethmaccrc32checker_packet_error <= 1'd0;
	end else begin
		if ((mac_core_liteethmaccrc32checker_sink_sink_valid & mac_core_liteethmaccrc32checker_sink_sink_ready)) begin
			mac_core_liteethmaccrc32checker_packet_error <= (mac_core_liteethmaccrc32checker_packet_error | ((mac_core_liteethmaccrc32checker_sink_sink_payload_error & mac_core_liteethmaccrc32checker_sink_sink_payload_be) != 1'd0));
		end
	end
	if (mac_core_liteethmaccrc32checker_crc_ce) begin
		mac_core_liteethmaccrc32checker_crc_reg <= mac_core_liteethmaccrc32checker_crc_crc_next;
	end
	if (mac_core_liteethmaccrc32checker_crc_reset) begin
		mac_core_liteethmaccrc32checker_crc_reg <= 32'd4294967295;
	end
	if (((mac_core_liteethmaccrc32checker_syncfifo_syncfifo_we & mac_core_liteethmaccrc32checker_syncfifo_syncfifo_writable) & (~mac_core_liteethmaccrc32checker_syncfifo_replace))) begin
		if ((mac_core_liteethmaccrc32checker_syncfifo_produce == 3'd4)) begin
			mac_core_liteethmaccrc32checker_syncfifo_produce <= 1'd0;
		end else begin
			mac_core_liteethmaccrc32checker_syncfifo_produce <= (mac_core_liteethmaccrc32checker_syncfifo_produce + 1'd1);
		end
	end
	if (mac_core_liteethmaccrc32checker_syncfifo_do_read) begin
		if ((mac_core_liteethmaccrc32checker_syncfifo_consume == 3'd4)) begin
			mac_core_liteethmaccrc32checker_syncfifo_consume <= 1'd0;
		end else begin
			mac_core_liteethmaccrc32checker_syncfifo_consume <= (mac_core_liteethmaccrc32checker_syncfifo_consume + 1'd1);
		end
	end
	if (((mac_core_liteethmaccrc32checker_syncfifo_syncfifo_we & mac_core_liteethmaccrc32checker_syncfifo_syncfifo_writable) & (~mac_core_liteethmaccrc32checker_syncfifo_replace))) begin
		if ((~mac_core_liteethmaccrc32checker_syncfifo_do_read)) begin
			mac_core_liteethmaccrc32checker_syncfifo_level <= (mac_core_liteethmaccrc32checker_syncfifo_level + 1'd1);
		end
	end else begin
		if (mac_core_liteethmaccrc32checker_syncfifo_do_read) begin
			mac_core_liteethmaccrc32checker_syncfifo_level <= (mac_core_liteethmaccrc32checker_syncfifo_level - 1'd1);
		end
	end
	if (mac_core_liteethmaccrc32checker_fifo_reset) begin
		mac_core_liteethmaccrc32checker_syncfifo_level <= 3'd0;
		mac_core_liteethmaccrc32checker_syncfifo_produce <= 3'd0;
		mac_core_liteethmaccrc32checker_syncfifo_consume <= 3'd0;
	end
	liteethmac_rxdatapath_bufferizeendpoints_state <= liteethmac_rxdatapath_bufferizeendpoints_next_state;
	if (mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value_ce0) begin
		mac_core_liteethmaccrc32checker_be <= mac_core_liteethmaccrc32checker_be_liteethmaccore_next_value0;
	end
	if (mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value_ce1) begin
		mac_core_liteethmaccrc32checker_last_error <= mac_core_liteethmaccrc32checker_last_error_liteethmaccore_next_value1;
	end
	if (((~mac_core_bufferizeendpoints_pipe_valid_source_valid) | mac_core_bufferizeendpoints_pipe_valid_source_ready)) begin
		mac_core_bufferizeendpoints_pipe_valid_source_valid <= mac_core_bufferizeendpoints_pipe_valid_sink_valid;
		mac_core_bufferizeendpoints_pipe_valid_source_first <= mac_core_bufferizeendpoints_pipe_valid_sink_first;
		mac_core_bufferizeendpoints_pipe_valid_source_last <= mac_core_bufferizeendpoints_pipe_valid_sink_last;
		mac_core_bufferizeendpoints_pipe_valid_source_payload_data <= mac_core_bufferizeendpoints_pipe_valid_sink_payload_data;
		mac_core_bufferizeendpoints_pipe_valid_source_payload_be <= mac_core_bufferizeendpoints_pipe_valid_sink_payload_be;
		mac_core_bufferizeendpoints_pipe_valid_source_payload_error <= mac_core_bufferizeendpoints_pipe_valid_sink_payload_error;
	end
	if (mac_core_pulsesynchronizer1_i) begin
		mac_core_pulsesynchronizer1_toggle_i <= (~mac_core_pulsesynchronizer1_toggle_i);
	end
	mac_core_pulsesynchronizer1_toggle_o_r <= mac_core_pulsesynchronizer1_toggle_o;
	if ((mac_core_rx_padding_sink_valid & mac_core_rx_padding_sink_ready)) begin
		if (mac_core_rx_padding_sink_last) begin
			mac_core_rx_padding_length <= 1'd0;
		end else begin
			mac_core_rx_padding_length <= (mac_core_rx_padding_length + mac_core_rx_padding_length_inc);
		end
	end
	case (liteethmac_grant)
		1'd0: begin
			if ((~liteethmac_request[0])) begin
				if (liteethmac_request[1]) begin
					liteethmac_grant <= 1'd1;
				end
			end
		end
		1'd1: begin
			if ((~liteethmac_request[1])) begin
				if (liteethmac_request[0]) begin
					liteethmac_grant <= 1'd0;
				end
			end
		end
	endcase
	liteethmac_status0_ongoing1 <= liteethmac_status0_ongoing0;
	if (liteethmac_status0_last) begin
		liteethmac_status0_first <= 1'd1;
	end else begin
		if ((arp_mac_port_sink_valid & arp_mac_port_sink_ready)) begin
			liteethmac_status0_first <= 1'd0;
		end
	end
	liteethmac_status1_ongoing1 <= liteethmac_status1_ongoing0;
	if (liteethmac_status1_last) begin
		liteethmac_status1_first <= 1'd1;
	end else begin
		if ((ip_mac_port_sink_valid & ip_mac_port_sink_ready)) begin
			liteethmac_status1_first <= 1'd0;
		end
	end
	if (liteethmac_last) begin
		liteethmac_sel_locked <= 1'd0;
	end else begin
		if (((liteethmac_first & mac_crossbar_sink_valid) & (~liteethmac_sel_locked))) begin
			liteethmac_sel_ongoing <= liteethmac_sel0;
			liteethmac_sel_locked <= 1'd1;
		end
	end
	liteethmac_ongoing1 <= liteethmac_ongoing0;
	if (liteethmac_last) begin
		liteethmac_first <= 1'd1;
	end else begin
		if ((mac_crossbar_sink_valid & mac_crossbar_sink_ready)) begin
			liteethmac_first <= 1'd0;
		end
	end
	liteethmac_liteethmacpacketizer_state <= liteethmac_liteethmacpacketizer_next_state;
	if (mac_packetizer_count_liteethmacpacketizer_next_value_ce) begin
		mac_packetizer_count <= mac_packetizer_count_liteethmacpacketizer_next_value;
	end
	if (((mac_depacketizer_sink_valid & mac_depacketizer_sink_ready) & (mac_depacketizer_skip_cnt < 1'd0))) begin
		mac_depacketizer_skip_cnt <= (mac_depacketizer_skip_cnt + 1'd1);
	end
	if (mac_depacketizer_sr_shift) begin
		mac_depacketizer_sr <= {mac_depacketizer_sink_payload_data, mac_depacketizer_sr[111:8]};
	end
	liteethmac_liteethmacdepacketizer_state <= liteethmac_liteethmacdepacketizer_next_state;
	if (mac_depacketizer_count_liteethmacdepacketizer_next_value_ce) begin
		mac_depacketizer_count <= mac_depacketizer_count_liteethmacdepacketizer_next_value;
	end
	if (mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value_ce0) begin
		mac_depacketizer_data_copy_first <= mac_depacketizer_data_copy_first_liteethmacdepacketizer_t_next_value0;
	end
	if (mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value_ce1) begin
		mac_depacketizer_send_dummy <= mac_depacketizer_send_dummy_liteethmacdepacketizer_t_next_value1;
	end
	if (mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value_ce) begin
		mac_depacketizer_skip_cnt <= mac_depacketizer_skip_cnt_liteethmacdepacketizer_f_next_value;
	end
	liteetharptx_liteetharppacketizer_state <= liteetharptx_liteetharppacketizer_next_state;
	if (arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value_ce) begin
		arp_tx_packetizer_count <= arp_tx_packetizer_count_liteetharp_liteetharptx_liteetharppacketizer_next_value;
	end
	liteetharptx_fsm_state <= liteetharptx_fsm_next_state;
	if (arp_tx_counter_liteetharp_liteetharptx_fsm_next_value_ce) begin
		arp_tx_counter <= arp_tx_counter_liteetharp_liteetharptx_fsm_next_value;
	end
	arp_rx_valid <= (((((arp_rx_depacketizer_source_valid & (arp_rx_depacketizer_source_param_hwtype == 1'd1)) & (arp_rx_depacketizer_source_param_proto == 12'd2048)) & (arp_rx_depacketizer_source_param_hwsize == 3'd6)) & (arp_rx_depacketizer_source_param_protosize == 3'd4)) & (arp_rx_depacketizer_source_param_target_ip == 32'd3232246274));
	if (((arp_rx_depacketizer_sink_valid & arp_rx_depacketizer_sink_ready) & (arp_rx_depacketizer_skip_cnt < 1'd0))) begin
		arp_rx_depacketizer_skip_cnt <= (arp_rx_depacketizer_skip_cnt + 1'd1);
	end
	if (arp_rx_depacketizer_sr_shift) begin
		arp_rx_depacketizer_sr <= {arp_rx_depacketizer_sink_payload_data, arp_rx_depacketizer_sr[223:8]};
	end
	liteetharprx_liteetharpdepacketizer_state <= liteetharprx_liteetharpdepacketizer_next_state;
	if (arp_rx_depacketizer_count_liteetharp_next_value_ce) begin
		arp_rx_depacketizer_count <= arp_rx_depacketizer_count_liteetharp_next_value;
	end
	if (arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value_ce0) begin
		arp_rx_depacketizer_data_copy_first <= arp_rx_depacketizer_data_copy_first_liteetharp_t_next_value0;
	end
	if (arp_rx_depacketizer_send_dummy_liteetharp_t_next_value_ce1) begin
		arp_rx_depacketizer_send_dummy <= arp_rx_depacketizer_send_dummy_liteetharp_t_next_value1;
	end
	if (arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value_ce) begin
		arp_rx_depacketizer_skip_cnt <= arp_rx_depacketizer_skip_cnt_liteetharp_f_next_value;
	end
	liteetharprx_fsm_state <= liteetharprx_fsm_next_state;
	if (arp_table_request_timer_wait) begin
		if ((~arp_table_request_timer_done)) begin
			arp_table_request_timer_count <= (arp_table_request_timer_count - 1'd1);
		end
	end else begin
		arp_table_request_timer_count <= 23'd5000000;
	end
	if (arp_table_cache_wait) begin
		if ((~arp_table_cache_done)) begin
			arp_table_cache_count <= (arp_table_cache_count - 1'd1);
		end
	end else begin
		arp_table_cache_count <= 26'd50000000;
	end
	liteetharptable_liteetharpcache_state <= liteetharptable_liteetharpcache_next_state;
	if (arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce0) begin
		arp_table_cache_update_count <= arp_table_cache_update_count_liteetharp_liteetharptable_liteetharpcache_next_value0;
	end
	if (arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value_ce1) begin
		arp_table_cache_search_count <= arp_table_cache_search_count_liteetharp_liteetharptable_liteetharpcache_next_value1;
	end
	if (arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value_ce2) begin
		arp_table_cache_error <= arp_table_cache_error_liteetharp_liteetharptable_liteetharpcache_next_value2;
	end
	liteetharptable_fsm_state <= liteetharptable_fsm_next_state;
	if (arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value_ce0) begin
		arp_table_request_pending <= arp_table_request_pending_liteetharp_liteetharptable_fsm_next_value0;
	end
	if (arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value_ce1) begin
		arp_table_response_response_payload_mac_address <= arp_table_response_response_payload_mac_address_liteetharp_liteetharptable_fsm_next_value1;
	end
	if (arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value_ce2) begin
		arp_table_response_response_payload_failed <= arp_table_response_response_payload_failed_liteetharp_liteetharptable_fsm_next_value2;
	end
	if (arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value_ce3) begin
		arp_table_request_counter <= arp_table_request_counter_liteetharp_liteetharptable_fsm_next_value3;
	end
	if (arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value_ce4) begin
		arp_table_request_ip_address <= arp_table_request_ip_address_liteetharp_liteetharptable_fsm_next_value4;
	end
	if (((~ip_tx_pipe_valid_source_valid) | ip_tx_pipe_valid_source_ready)) begin
		ip_tx_pipe_valid_source_valid <= ip_tx_pipe_valid_sink_valid;
		ip_tx_pipe_valid_source_first <= ip_tx_pipe_valid_sink_first;
		ip_tx_pipe_valid_source_last <= ip_tx_pipe_valid_sink_last;
		ip_tx_pipe_valid_source_payload_data <= ip_tx_pipe_valid_sink_payload_data;
		ip_tx_pipe_valid_source_payload_be <= ip_tx_pipe_valid_sink_payload_be;
		ip_tx_pipe_valid_source_payload_error <= ip_tx_pipe_valid_sink_payload_error;
		ip_tx_pipe_valid_source_param_length <= ip_tx_pipe_valid_sink_param_length;
		ip_tx_pipe_valid_source_param_protocol <= ip_tx_pipe_valid_sink_param_protocol;
		ip_tx_pipe_valid_source_param_ip_address <= ip_tx_pipe_valid_sink_param_ip_address;
	end
	if (ip_tx_ce) begin
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next0 <= {ip_tx_liteethipv4checksum0, (ip_tx_liteethipv4checksum_s_next0[15:0] + ip_tx_liteethipv4checksum_s_next0[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next1 <= {ip_tx_liteethipv4checksum1, (ip_tx_liteethipv4checksum_s_next1[15:0] + ip_tx_liteethipv4checksum_s_next1[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next2 <= {ip_tx_liteethipv4checksum2, (ip_tx_liteethipv4checksum_s_next2[15:0] + ip_tx_liteethipv4checksum_s_next2[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next3 <= {ip_tx_liteethipv4checksum3, (ip_tx_liteethipv4checksum_s_next3[15:0] + ip_tx_liteethipv4checksum_s_next3[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next4 <= {ip_tx_liteethipv4checksum4, (ip_tx_liteethipv4checksum_s_next4[15:0] + ip_tx_liteethipv4checksum_s_next4[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next5 <= {ip_tx_liteethipv4checksum5, (ip_tx_liteethipv4checksum_s_next5[15:0] + ip_tx_liteethipv4checksum_s_next5[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next6 <= {ip_tx_liteethipv4checksum6, (ip_tx_liteethipv4checksum_s_next6[15:0] + ip_tx_liteethipv4checksum_s_next6[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next7 <= {ip_tx_liteethipv4checksum7, (ip_tx_liteethipv4checksum_s_next7[15:0] + ip_tx_liteethipv4checksum_s_next7[16])};
		end
		if ((~ip_tx_liteethipv4checksum_done)) begin
			ip_tx_liteethipv4checksum_r_next8 <= {ip_tx_liteethipv4checksum8, (ip_tx_liteethipv4checksum_s_next8[15:0] + ip_tx_liteethipv4checksum_s_next8[16])};
		end
		if (ip_tx_liteethipv4checksum_counter_ce) begin
			ip_tx_liteethipv4checksum_counter <= (ip_tx_liteethipv4checksum_counter + 1'd1);
		end
	end
	if (ip_tx_reset) begin
		ip_tx_liteethipv4checksum_counter <= 4'd0;
	end
	liteethip_liteethiptx_liteethipv4packetizer_state <= liteethip_liteethiptx_liteethipv4packetizer_next_state;
	if (ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value_ce) begin
		ip_tx_packetizer_count <= ip_tx_packetizer_count_liteethip_liteethipv4packetizer_next_value;
	end
	liteethip_liteethiptx_fsm_state <= liteethip_liteethiptx_fsm_next_state;
	if (ip_tx_target_mac_liteethip_fsm_next_value_ce) begin
		ip_tx_target_mac <= ip_tx_target_mac_liteethip_fsm_next_value;
	end
	if (((ip_rx_depacketizer_sink_valid & ip_rx_depacketizer_sink_ready) & (ip_rx_depacketizer_skip_cnt < 1'd0))) begin
		ip_rx_depacketizer_skip_cnt <= (ip_rx_depacketizer_skip_cnt + 1'd1);
	end
	if (ip_rx_depacketizer_sr_shift) begin
		ip_rx_depacketizer_sr <= {ip_rx_depacketizer_sink_payload_data, ip_rx_depacketizer_sr[159:8]};
	end
	liteethip_liteethiprx_liteethipv4depacketizer_state <= liteethip_liteethiprx_liteethipv4depacketizer_next_state;
	if (ip_rx_depacketizer_count_liteethip_next_value_ce) begin
		ip_rx_depacketizer_count <= ip_rx_depacketizer_count_liteethip_next_value;
	end
	if (ip_rx_depacketizer_data_copy_first_liteethip_t_next_value_ce0) begin
		ip_rx_depacketizer_data_copy_first <= ip_rx_depacketizer_data_copy_first_liteethip_t_next_value0;
	end
	if (ip_rx_depacketizer_send_dummy_liteethip_t_next_value_ce1) begin
		ip_rx_depacketizer_send_dummy <= ip_rx_depacketizer_send_dummy_liteethip_t_next_value1;
	end
	if (ip_rx_depacketizer_skip_cnt_liteethip_f_next_value_ce) begin
		ip_rx_depacketizer_skip_cnt <= ip_rx_depacketizer_skip_cnt_liteethip_f_next_value;
	end
	if (ip_rx_ce) begin
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next0 <= {ip_rx_liteethipv4checksum0, (ip_rx_liteethipv4checksum_s_next0[15:0] + ip_rx_liteethipv4checksum_s_next0[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next1 <= {ip_rx_liteethipv4checksum1, (ip_rx_liteethipv4checksum_s_next1[15:0] + ip_rx_liteethipv4checksum_s_next1[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next2 <= {ip_rx_liteethipv4checksum2, (ip_rx_liteethipv4checksum_s_next2[15:0] + ip_rx_liteethipv4checksum_s_next2[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next3 <= {ip_rx_liteethipv4checksum3, (ip_rx_liteethipv4checksum_s_next3[15:0] + ip_rx_liteethipv4checksum_s_next3[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next4 <= {ip_rx_liteethipv4checksum4, (ip_rx_liteethipv4checksum_s_next4[15:0] + ip_rx_liteethipv4checksum_s_next4[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next5 <= {ip_rx_liteethipv4checksum5, (ip_rx_liteethipv4checksum_s_next5[15:0] + ip_rx_liteethipv4checksum_s_next5[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next6 <= {ip_rx_liteethipv4checksum6, (ip_rx_liteethipv4checksum_s_next6[15:0] + ip_rx_liteethipv4checksum_s_next6[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next7 <= {ip_rx_liteethipv4checksum7, (ip_rx_liteethipv4checksum_s_next7[15:0] + ip_rx_liteethipv4checksum_s_next7[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next8 <= {ip_rx_liteethipv4checksum8, (ip_rx_liteethipv4checksum_s_next8[15:0] + ip_rx_liteethipv4checksum_s_next8[16])};
		end
		if ((~ip_rx_liteethipv4checksum_done)) begin
			ip_rx_liteethipv4checksum_r_next9 <= {ip_rx_liteethipv4checksum9, (ip_rx_liteethipv4checksum_s_next9[15:0] + ip_rx_liteethipv4checksum_s_next9[16])};
		end
		if (ip_rx_liteethipv4checksum_counter_ce) begin
			ip_rx_liteethipv4checksum_counter <= (ip_rx_liteethipv4checksum_counter + 1'd1);
		end
	end
	if (ip_rx_reset) begin
		ip_rx_liteethipv4checksum_counter <= 4'd0;
	end
	liteethip_liteethiprx_fsm_state <= liteethip_liteethiprx_fsm_next_state;
	if (((~ip_rx_buffer_pipe_valid_source_valid) | ip_rx_buffer_pipe_valid_source_ready)) begin
		ip_rx_buffer_pipe_valid_source_valid <= ip_rx_buffer_pipe_valid_sink_valid;
		ip_rx_buffer_pipe_valid_source_first <= ip_rx_buffer_pipe_valid_sink_first;
		ip_rx_buffer_pipe_valid_source_last <= ip_rx_buffer_pipe_valid_sink_last;
		ip_rx_buffer_pipe_valid_source_payload_data <= ip_rx_buffer_pipe_valid_sink_payload_data;
		ip_rx_buffer_pipe_valid_source_payload_be <= ip_rx_buffer_pipe_valid_sink_payload_be;
		ip_rx_buffer_pipe_valid_source_payload_error <= ip_rx_buffer_pipe_valid_sink_payload_error;
		ip_rx_buffer_pipe_valid_source_param_length <= ip_rx_buffer_pipe_valid_sink_param_length;
		ip_rx_buffer_pipe_valid_source_param_protocol <= ip_rx_buffer_pipe_valid_sink_param_protocol;
		ip_rx_buffer_pipe_valid_source_param_ip_address <= ip_rx_buffer_pipe_valid_sink_param_ip_address;
	end
	if ((ip_rx_buffer_pipe_ready_sink_valid & (~ip_rx_buffer_pipe_ready_source_ready))) begin
		ip_rx_buffer_pipe_ready_valid <= 1'd1;
	end else begin
		if (ip_rx_buffer_pipe_ready_source_ready) begin
			ip_rx_buffer_pipe_ready_valid <= 1'd0;
		end
	end
	if ((~ip_rx_buffer_pipe_ready_valid)) begin
		ip_rx_buffer_pipe_ready_sink_d_valid <= ip_rx_buffer_pipe_ready_sink_valid;
		ip_rx_buffer_pipe_ready_sink_d_ready <= ip_rx_buffer_pipe_ready_sink_ready;
		ip_rx_buffer_pipe_ready_sink_d_first <= ip_rx_buffer_pipe_ready_sink_first;
		ip_rx_buffer_pipe_ready_sink_d_last <= ip_rx_buffer_pipe_ready_sink_last;
		ip_rx_buffer_pipe_ready_sink_d_payload_data <= ip_rx_buffer_pipe_ready_sink_payload_data;
		ip_rx_buffer_pipe_ready_sink_d_payload_be <= ip_rx_buffer_pipe_ready_sink_payload_be;
		ip_rx_buffer_pipe_ready_sink_d_payload_error <= ip_rx_buffer_pipe_ready_sink_payload_error;
		ip_rx_buffer_pipe_ready_sink_d_param_length <= ip_rx_buffer_pipe_ready_sink_param_length;
		ip_rx_buffer_pipe_ready_sink_d_param_protocol <= ip_rx_buffer_pipe_ready_sink_param_protocol;
		ip_rx_buffer_pipe_ready_sink_d_param_ip_address <= ip_rx_buffer_pipe_ready_sink_param_ip_address;
	end
	case (liteethip_grant)
		1'd0: begin
			if ((~liteethip_request[0])) begin
				if (liteethip_request[1]) begin
					liteethip_grant <= 1'd1;
				end
			end
		end
		1'd1: begin
			if ((~liteethip_request[1])) begin
				if (liteethip_request[0]) begin
					liteethip_grant <= 1'd0;
				end
			end
		end
	endcase
	liteethip_status0_ongoing1 <= liteethip_status0_ongoing0;
	if (liteethip_status0_last) begin
		liteethip_status0_first <= 1'd1;
	end else begin
		if ((icmp_sink_valid & icmp_sink_ready)) begin
			liteethip_status0_first <= 1'd0;
		end
	end
	liteethip_status1_ongoing1 <= liteethip_status1_ongoing0;
	if (liteethip_status1_last) begin
		liteethip_status1_first <= 1'd1;
	end else begin
		if ((ip_port_sink_valid & ip_port_sink_ready)) begin
			liteethip_status1_first <= 1'd0;
		end
	end
	if (liteethip_last) begin
		liteethip_sel_locked <= 1'd0;
	end else begin
		if (((liteethip_first & ip_crossbar_sink_valid) & (~liteethip_sel_locked))) begin
			liteethip_sel_ongoing <= liteethip_sel0;
			liteethip_sel_locked <= 1'd1;
		end
	end
	liteethip_ongoing1 <= liteethip_ongoing0;
	if (liteethip_last) begin
		liteethip_first <= 1'd1;
	end else begin
		if ((ip_crossbar_sink_valid & ip_crossbar_sink_ready)) begin
			liteethip_first <= 1'd0;
		end
	end
	if (icmp_tx_source_source_valid) begin
		if ((~icmp_tx_active)) begin
			icmp_tx_length <= (icmp_tx_sink_sink_param_length + 4'd8);
			icmp_tx_ip_address <= icmp_tx_sink_sink_param_ip_address;
			icmp_tx_active <= 1'd1;
		end
		if ((icmp_tx_source_source_ready & icmp_tx_source_source_last)) begin
			icmp_tx_active <= 1'd0;
		end
	end
	liteethicmp_state <= liteethicmp_next_state;
	if (icmp_tx_packetizer_count_liteethicmp_next_value_ce) begin
		icmp_tx_packetizer_count <= icmp_tx_packetizer_count_liteethicmp_next_value;
	end
	if ((icmp_rx_sink_sink_valid & icmp_rx_sink_sink_ready)) begin
		if ((~icmp_rx_active)) begin
			icmp_rx_length <= icmp_rx_sink_sink_param_length;
			icmp_rx_protocol <= icmp_rx_sink_sink_param_protocol;
			icmp_rx_ip_address <= icmp_rx_sink_sink_param_ip_address;
		end
		icmp_rx_active <= (~icmp_rx_sink_sink_last);
	end
	if (icmp_rx_depacketizer_sr_shift) begin
		icmp_rx_depacketizer_sr <= {icmp_rx_depacketizer_sink_payload_data, icmp_rx_depacketizer_sr[63:8]};
	end
	if (icmp_rx_depacketizer_sr_shift_leftover) begin
		icmp_rx_depacketizer_sr <= {icmp_rx_depacketizer_sink_payload_data, icmp_rx_depacketizer_sr[63:0]};
	end
	liteethicmp_liteethicmpdepacketizer_state <= liteethicmp_liteethicmpdepacketizer_next_state;
	if (icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value_ce0) begin
		icmp_rx_depacketizer_count <= icmp_rx_depacketizer_count_liteethicmp_liteethicmpdepacketizer_next_value0;
	end
	if (icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value_ce1) begin
		icmp_rx_depacketizer_fsm_from_idle <= icmp_rx_depacketizer_fsm_from_idle_liteethicmp_liteethicmpdepacketizer_next_value1;
	end
	if (icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value_ce2) begin
		icmp_rx_depacketizer_data_copy_first <= icmp_rx_depacketizer_data_copy_first_liteethicmp_liteethicmpdepacketizer_next_value2;
	end
	liteethicmp_fsm_state <= liteethicmp_fsm_next_state;
	if (icmp_rx_count_liteethicmp_fsm_next_value_ce) begin
		icmp_rx_count <= icmp_rx_count_liteethicmp_fsm_next_value;
	end
	if ((icmp_echo_sink_sink_valid & icmp_echo_sink_sink_ready)) begin
		icmp_echo_busy <= 1'd1;
		icmp_echo_receiving <= (~icmp_echo_sink_sink_last);
	end
	if ((icmp_echo_buffer_drop | ((icmp_echo_source_source_valid & icmp_echo_source_source_ready) & icmp_echo_source_source_last))) begin
		icmp_echo_busy <= 1'd0;
	end
	if ((icmp_echo_sink_sink_valid & icmp_echo_sink_sink_ready)) begin
		if (icmp_echo_sink_sink_last) begin
			icmp_echo_count <= 1'd0;
		end else begin
			if ((icmp_echo_count <= 8'd128)) begin
				icmp_echo_count <= icmp_echo_size;
			end
		end
	end
	if (icmp_echo_buffer_sink_valid) begin
		icmp_echo_buffer_invalid <= (icmp_echo_buffer_invalid | icmp_echo_buffer_discard);
		if (((~icmp_echo_buffer_overflow) & (~icmp_echo_buffer_full))) begin
			icmp_echo_buffer_wr_ptr <= (icmp_echo_buffer_wr_ptr + 1'd1);
		end else begin
			icmp_echo_buffer_overflow <= 1'd1;
		end
		if (icmp_echo_buffer_sink_last) begin
			icmp_echo_buffer_overflow <= 1'd0;
			icmp_echo_buffer_invalid <= 1'd0;
			if (icmp_echo_buffer_commit) begin
				icmp_echo_buffer_commit_ptr <= (icmp_echo_buffer_wr_ptr + 1'd1);
			end else begin
				icmp_echo_buffer_wr_ptr <= icmp_echo_buffer_commit_ptr;
			end
		end
	end
	if (icmp_echo_buffer_fetch) begin
		icmp_echo_buffer_rd_ptr <= (icmp_echo_buffer_rd_ptr + 1'd1);
		icmp_echo_buffer_out_valid <= 1'd1;
	end else begin
		if ((icmp_echo_buffer_source_valid & icmp_echo_buffer_source_ready)) begin
			icmp_echo_buffer_out_valid <= 1'd0;
		end
	end
	if (((~icmp_echo_buffer_pipe_valid_source_valid) | icmp_echo_buffer_pipe_valid_source_ready)) begin
		icmp_echo_buffer_pipe_valid_source_valid <= icmp_echo_buffer_pipe_valid_sink_valid;
		icmp_echo_buffer_pipe_valid_source_first <= icmp_echo_buffer_pipe_valid_sink_first;
		icmp_echo_buffer_pipe_valid_source_last <= icmp_echo_buffer_pipe_valid_sink_last;
		icmp_echo_buffer_pipe_valid_source_param_checksum <= icmp_echo_buffer_pipe_valid_sink_param_checksum;
		icmp_echo_buffer_pipe_valid_source_param_code <= icmp_echo_buffer_pipe_valid_sink_param_code;
		icmp_echo_buffer_pipe_valid_source_param_msgtype <= icmp_echo_buffer_pipe_valid_sink_param_msgtype;
		icmp_echo_buffer_pipe_valid_source_param_quench <= icmp_echo_buffer_pipe_valid_sink_param_quench;
		icmp_echo_buffer_pipe_valid_source_param_ip_address <= icmp_echo_buffer_pipe_valid_sink_param_ip_address;
		icmp_echo_buffer_pipe_valid_source_param_length <= icmp_echo_buffer_pipe_valid_sink_param_length;
	end
	if (tx_source_source_valid) begin
		if ((~tx_active)) begin
			tx_length <= tx_packetizer_sink_param_length;
			tx_ip_address <= tx_sink_sink_param_ip_address;
			tx_active <= 1'd1;
		end
		if ((tx_source_source_ready & tx_source_source_last)) begin
			tx_active <= 1'd0;
		end
	end
	liteethudp_state <= liteethudp_next_state;
	if (tx_packetizer_count_liteethudp_next_value_ce) begin
		tx_packetizer_count <= tx_packetizer_count_liteethudp_next_value;
	end
	if ((rx_sink_sink_valid & rx_sink_sink_ready)) begin
		if ((~rx_active)) begin
			rx_length <= rx_sink_sink_param_length;
			rx_protocol <= rx_sink_sink_param_protocol;
			rx_ip_address <= rx_sink_sink_param_ip_address;
		end
		rx_active <= (~rx_sink_sink_last);
	end
	if (((rx_depacketizer_sink_valid & rx_depacketizer_sink_ready) & (rx_depacketizer_skip_cnt < 1'd0))) begin
		rx_depacketizer_skip_cnt <= (rx_depacketizer_skip_cnt + 1'd1);
	end
	if (rx_depacketizer_sr_shift) begin
		rx_depacketizer_sr <= {rx_depacketizer_sink_payload_data, rx_depacketizer_sr[63:8]};
	end
	liteethudp_liteethudpdepacketizer_state <= liteethudp_liteethudpdepacketizer_next_state;
	if (rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value_ce) begin
		rx_depacketizer_count <= rx_depacketizer_count_liteethudp_liteethudpdepacketizer_next_value;
	end
	if (rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value_ce0) begin
		rx_depacketizer_data_copy_first <= rx_depacketizer_data_copy_first_liteethudp_liteethudpdepacketizer_t_next_value0;
	end
	if (rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value_ce1) begin
		rx_depacketizer_send_dummy <= rx_depacketizer_send_dummy_liteethudp_liteethudpdepacketizer_t_next_value1;
	end
	if (rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value_ce) begin
		rx_depacketizer_skip_cnt <= rx_depacketizer_skip_cnt_liteethudp_liteethudpdepacketizer_f_next_value;
	end
	liteethudp_fsm_state <= liteethudp_fsm_next_state;
	if (rx_count_liteethudp_fsm_next_value_ce) begin
		rx_count <= rx_count_liteethudp_fsm_next_value;
	end
	if (liteethudp_last) begin
		liteethudp_sel_locked <= 1'd0;
	end else begin
		if (((liteethudp_first & crossbar_sink_valid) & (~liteethudp_sel_locked))) begin
			liteethudp_sel_ongoing <= liteethudp_sel0;
			liteethudp_sel_locked <= 1'd1;
		end
	end
	liteethudp_ongoing1 <= liteethudp_ongoing0;
	if (liteethudp_last) begin
		liteethudp_first <= 1'd1;
	end else begin
		if ((crossbar_sink_valid & crossbar_sink_ready)) begin
			liteethudp_first <= 1'd0;
		end
	end
	if (echo_buffer_syncfifo_re) begin
		echo_buffer_readable <= 1'd1;
	end else begin
		if (echo_buffer_re) begin
			echo_buffer_readable <= 1'd0;
		end
	end
	if (((echo_buffer_syncfifo_we & echo_buffer_syncfifo_writable) & (~echo_buffer_replace))) begin
		echo_buffer_produce <= (echo_buffer_produce + 1'd1);
	end
	if (echo_buffer_do_read) begin
		echo_buffer_consume <= (echo_buffer_consume + 1'd1);
	end
	if (((echo_buffer_syncfifo_we & echo_buffer_syncfifo_writable) & (~echo_buffer_replace))) begin
		if ((~echo_buffer_do_read)) begin
			echo_buffer_level0 <= (echo_buffer_level0 + 1'd1);
		end
	end else begin
		if (echo_buffer_do_read) begin
			echo_buffer_level0 <= (echo_buffer_level0 - 1'd1);
		end
	end
	if ((phy_source_valid & phy_source_last)) begin
		if ((value0 != 8'd255)) begin
			value0 <= (value0 + 1'd1);
		end
	end
	if ((mac_core_rxdatapath_source_valid & mac_core_rxdatapath_source_last)) begin
		if ((value1 != 8'd255)) begin
			value1 <= (value1 + 1'd1);
		end
	end
	if ((mac_depacketizer_sink_valid & mac_depacketizer_sink_last)) begin
		if ((value2 != 8'd255)) begin
			value2 <= (value2 + 1'd1);
		end
	end
	if ((arp_table_sink_valid & arp_table_sink_payload_request)) begin
		if ((value3 != 8'd255)) begin
			value3 <= (value3 + 1'd1);
		end
	end
	if (arp_table_source_valid) begin
		if ((value4 != 8'd255)) begin
			value4 <= (value4 + 1'd1);
		end
	end
	if ((phy_sink_valid & phy_sink_last)) begin
		if ((value5 != 8'd255)) begin
			value5 <= (value5 + 1'd1);
		end
	end
	if ((phy_source_valid & phy_source_ready)) begin
		if ((value6 != 8'd255)) begin
			value6 <= (value6 + 1'd1);
		end
	end
	if ((phy_sink_valid & phy_sink_ready)) begin
		if ((value7 != 8'd255)) begin
			value7 <= (value7 + 1'd1);
		end
	end
	if ((uarttx_state == 1'd0)) begin
		if (uarttx_strobe) begin
			uarttx_state <= 1'd1;
			uarttx_cnt <= 1'd0;
			uarttx_shreg <= uarttx_data;
			uarttx_bitidx <= 1'd0;
		end else begin
			if (uarttx_pend_bit) begin
				uarttx_state <= 1'd1;
				uarttx_cnt <= 1'd0;
				uarttx_shreg <= uarttx_pend_data;
				uarttx_pend_bit <= 1'd0;
				uarttx_bitidx <= 1'd0;
			end
		end
	end else begin
		if ((uarttx_state == 1'd1)) begin
			uarttx_cnt <= (uarttx_cnt + 1'd1);
			if ((uarttx_cnt == 9'd433)) begin
				uarttx_state <= 2'd2;
				uarttx_cnt <= 1'd0;
			end
		end else begin
			if ((uarttx_state == 2'd2)) begin
				uarttx_cnt <= (uarttx_cnt + 1'd1);
				if ((uarttx_cnt == 9'd433)) begin
					uarttx_shreg <= {uarttx, uarttx_shreg[7:1]};
					if ((uarttx_bitidx == 3'd7)) begin
						uarttx_state <= 2'd3;
					end else begin
						uarttx_bitidx <= (uarttx_bitidx + 1'd1);
					end
					uarttx_cnt <= 1'd0;
				end
			end else begin
				uarttx_cnt <= (uarttx_cnt + 1'd1);
				if ((uarttx_cnt == 9'd433)) begin
					uarttx_state <= 1'd0;
					uarttx_cnt <= 1'd0;
				end
			end
		end
	end
	if ((uarttx_strobe & (uarttx_state != 1'd0))) begin
		uarttx_pend_bit <= 1'd1;
		uarttx_pend_data <= uarttx_data;
	end
	if (sys_rst) begin
		mac_core_tx_padding_counter <= 16'd0;
		mac_core_tx_crc_reg <= 32'd4294967295;
		mac_core_tx_crc_cnt <= 2'd3;
		mac_core_tx_crc_pipe_valid_source_valid <= 1'd0;
		mac_core_preamble_errors_status <= 32'd0;
		mac_core_crc_errors_status <= 32'd0;
		mac_core_liteethmaccrc32checker_crc_reg <= 32'd4294967295;
		mac_core_liteethmaccrc32checker_syncfifo_level <= 3'd0;
		mac_core_liteethmaccrc32checker_syncfifo_produce <= 3'd0;
		mac_core_liteethmaccrc32checker_syncfifo_consume <= 3'd0;
		mac_core_liteethmaccrc32checker_last_error <= 1'd0;
		mac_core_liteethmaccrc32checker_packet_error <= 1'd0;
		mac_core_bufferizeendpoints_pipe_valid_source_valid <= 1'd0;
		mac_core_rx_padding_length <= 11'd0;
		mac_packetizer_count <= 4'd0;
		mac_depacketizer_count <= 4'd0;
		mac_depacketizer_skip_cnt <= 1'd0;
		mac_depacketizer_data_copy_first <= 1'd1;
		mac_depacketizer_send_dummy <= 1'd0;
		arp_tx_packetizer_count <= 5'd0;
		arp_rx_depacketizer_count <= 5'd0;
		arp_rx_depacketizer_skip_cnt <= 1'd0;
		arp_rx_depacketizer_data_copy_first <= 1'd1;
		arp_rx_depacketizer_send_dummy <= 1'd0;
		arp_table_request_pending <= 1'd0;
		arp_table_request_counter <= 4'd0;
		arp_table_request_timer_count <= 23'd5000000;
		arp_table_cache_update_count <= 1'd0;
		arp_table_cache_search_count <= 1'd0;
		arp_table_cache_error <= 1'd0;
		arp_table_cache_count <= 26'd50000000;
		ip_tx_pipe_valid_source_valid <= 1'd0;
		ip_tx_liteethipv4checksum_counter <= 4'd0;
		ip_tx_packetizer_count <= 5'd0;
		ip_rx_depacketizer_count <= 5'd0;
		ip_rx_depacketizer_skip_cnt <= 1'd0;
		ip_rx_depacketizer_data_copy_first <= 1'd1;
		ip_rx_depacketizer_send_dummy <= 1'd0;
		ip_rx_liteethipv4checksum_counter <= 4'd0;
		ip_rx_buffer_pipe_valid_source_valid <= 1'd0;
		ip_rx_buffer_pipe_ready_valid <= 1'd0;
		icmp_tx_packetizer_count <= 3'd0;
		icmp_tx_active <= 1'd0;
		icmp_tx_length <= 16'd0;
		icmp_tx_ip_address <= 32'd0;
		icmp_rx_depacketizer_count <= 3'd0;
		icmp_rx_depacketizer_data_copy_first <= 1'd0;
		icmp_rx_depacketizer_fsm_from_idle <= 1'd0;
		icmp_rx_active <= 1'd0;
		icmp_rx_length <= 16'd0;
		icmp_rx_protocol <= 8'd0;
		icmp_rx_ip_address <= 32'd0;
		icmp_rx_count <= 17'd0;
		icmp_echo_buffer_pipe_valid_source_valid <= 1'd0;
		icmp_echo_buffer_wr_ptr <= 8'd0;
		icmp_echo_buffer_commit_ptr <= 8'd0;
		icmp_echo_buffer_rd_ptr <= 8'd0;
		icmp_echo_buffer_out_valid <= 1'd0;
		icmp_echo_buffer_overflow <= 1'd0;
		icmp_echo_buffer_invalid <= 1'd0;
		icmp_echo_busy <= 1'd0;
		icmp_echo_receiving <= 1'd0;
		icmp_echo_count <= 8'd0;
		tx_packetizer_count <= 3'd0;
		tx_active <= 1'd0;
		tx_length <= 16'd0;
		tx_ip_address <= 32'd0;
		rx_depacketizer_count <= 3'd0;
		rx_depacketizer_skip_cnt <= 1'd0;
		rx_depacketizer_data_copy_first <= 1'd1;
		rx_depacketizer_send_dummy <= 1'd0;
		rx_active <= 1'd0;
		rx_length <= 16'd0;
		rx_protocol <= 8'd0;
		rx_ip_address <= 32'd0;
		rx_count <= 17'd0;
		echo_buffer_readable <= 1'd0;
		echo_buffer_level0 <= 9'd0;
		echo_buffer_produce <= 8'd0;
		echo_buffer_consume <= 8'd0;
		hold1 <= 8'd0;
		hold2 <= 8'd0;
		hold3 <= 8'd0;
		hold4 <= 8'd0;
		hold5 <= 8'd0;
		value0 <= 8'd0;
		value1 <= 8'd0;
		value2 <= 8'd0;
		value3 <= 8'd0;
		value4 <= 8'd0;
		value5 <= 8'd0;
		value6 <= 8'd0;
		value7 <= 8'd0;
		uarttx_state <= 2'd0;
		uarttx_cnt <= 10'd0;
		uarttx_shreg <= 8'd0;
		uarttx_bitidx <= 3'd0;
		uarttx_pend_bit <= 1'd0;
		uarttx_pend_data <= 8'd0;
		timer <= 23'd8388607;
		fire <= 1'd0;
		idx <= 4'd0;
		dumping <= 1'd0;
		liteethmac_txdatapath_liteethmacpaddinginserter_state <= 1'd0;
		liteethmac_txdatapath_bufferizeendpoints_state <= 2'd0;
		liteethmac_txdatapath_liteethmacpreambleinserter_state <= 2'd0;
		liteethmac_rxdatapath_liteethmacpreamblechecker_state <= 1'd0;
		liteethmac_rxdatapath_bufferizeendpoints_state <= 2'd0;
		liteethmac_grant <= 1'd0;
		liteethmac_status0_first <= 1'd1;
		liteethmac_status0_ongoing1 <= 1'd0;
		liteethmac_status1_first <= 1'd1;
		liteethmac_status1_ongoing1 <= 1'd0;
		liteethmac_first <= 1'd1;
		liteethmac_ongoing1 <= 1'd0;
		liteethmac_sel_ongoing <= 2'd0;
		liteethmac_sel_locked <= 1'd0;
		liteethmac_liteethmacpacketizer_state <= 1'd1;
		liteethmac_liteethmacdepacketizer_state <= 2'd0;
		liteetharptx_liteetharppacketizer_state <= 1'd1;
		liteetharptx_fsm_state <= 1'd0;
		liteetharprx_liteetharpdepacketizer_state <= 2'd0;
		liteetharprx_fsm_state <= 2'd0;
		liteetharptable_liteetharpcache_state <= 3'd0;
		liteetharptable_fsm_state <= 3'd0;
		liteethip_liteethiptx_liteethipv4packetizer_state <= 1'd1;
		liteethip_liteethiptx_fsm_state <= 3'd2;
		liteethip_liteethiprx_liteethipv4depacketizer_state <= 2'd0;
		liteethip_liteethiprx_fsm_state <= 2'd1;
		liteethip_grant <= 1'd0;
		liteethip_status0_first <= 1'd1;
		liteethip_status0_ongoing1 <= 1'd0;
		liteethip_status1_first <= 1'd1;
		liteethip_status1_ongoing1 <= 1'd0;
		liteethip_first <= 1'd1;
		liteethip_ongoing1 <= 1'd0;
		liteethip_sel_ongoing <= 2'd0;
		liteethip_sel_locked <= 1'd0;
		liteethicmp_state <= 1'd1;
		liteethicmp_liteethicmpdepacketizer_state <= 2'd0;
		liteethicmp_fsm_state <= 2'd2;
		liteethudp_state <= 1'd1;
		liteethudp_liteethudpdepacketizer_state <= 2'd0;
		liteethudp_fsm_state <= 2'd2;
		liteethudp_first <= 1'd1;
		liteethudp_ongoing1 <= 1'd0;
		liteethudp_sel_ongoing <= 1'd0;
		liteethudp_sel_locked <= 1'd0;
	end
	multiregimpl00 <= mac_core_pulsesynchronizer0_toggle_i;
	multiregimpl01 <= multiregimpl00;
	multiregimpl10 <= mac_core_pulsesynchronizer1_toggle_i;
	multiregimpl11 <= multiregimpl10;
end

reg [11:0] storage[0:4];
always @(posedge sys_clk) begin : mem_write_block
	if (mac_core_liteethmaccrc32checker_syncfifo_wrport_we)
		storage[mac_core_liteethmaccrc32checker_syncfifo_wrport_adr] <= mac_core_liteethmaccrc32checker_syncfifo_wrport_dat_w;
end

always @(posedge sys_clk) begin : mem_write_block_1
end

assign mac_core_liteethmaccrc32checker_syncfifo_rdport_dat_r = storage[mac_core_liteethmaccrc32checker_syncfifo_rdport_adr];

reg [80:0] mem[0:1];
reg [0:0] memadr;
always @(posedge sys_clk) begin : mem_write_block_2
	if (arp_table_cache_mem_wr_port_we)
		mem[arp_table_cache_mem_wr_port_adr] <= arp_table_cache_mem_wr_port_dat_w;
	memadr <= arp_table_cache_mem_wr_port_adr;
end

always @(posedge sys_clk) begin : mem_write_block_3
end

assign arp_table_cache_mem_wr_port_dat_r = mem[memadr];
assign arp_table_cache_mem_rd_port_dat_r = mem[arp_table_cache_mem_rd_port_adr];

reg [11:0] mem_1[0:127];
reg [6:0] memadr_1;
reg [6:0] memadr_2;
always @(posedge sys_clk) begin : mem_write_block_4
	if (icmp_echo_buffer_wr_port_we)
		mem_1[icmp_echo_buffer_wr_port_adr] <= icmp_echo_buffer_wr_port_dat_w;
	memadr_1 <= icmp_echo_buffer_wr_port_adr;
end

always @(posedge sys_clk) begin : mem_write_block_5
	if (icmp_echo_buffer_rd_port_re)
		memadr_2 <= icmp_echo_buffer_rd_port_adr;
end

assign icmp_echo_buffer_wr_port_dat_r = mem_1[memadr_1];
assign icmp_echo_buffer_rd_port_dat_r = mem_1[memadr_2];

reg [91:0] storage_1[0:255];
reg [91:0] memdat;
always @(posedge sys_clk) begin : mem_write_block_6
	if (echo_buffer_wrport_we)
		storage_1[echo_buffer_wrport_adr] <= echo_buffer_wrport_dat_w;
end

always @(posedge sys_clk) begin : mem_write_block_7
	if (echo_buffer_rdport_re)
		memdat <= storage_1[echo_buffer_rdport_adr];
end

assign echo_buffer_rdport_dat_r = memdat;

endmodule
